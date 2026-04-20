"""
preprocessing.py
================
Digital Image Processing pipeline for Brain Tumor MRI Classification.

Pipeline stages
---------------
1.  Load & validate image
2.  Grayscale conversion + CLAHE contrast enhancement
3.  Noise suppression (Gaussian + median)
4.  Skull-stripping / brain extraction (morphological + contour)
5.  Tumor Region-of-Interest segmentation (Otsu + morphology)
6.  Feature-aware resize to model input size
7.  Augmentation helpers (training only)

Author: NeuroScan AI
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & VALIDATE
# ──────────────────────────────────────────────────────────────────────────────

def load_image(source) -> np.ndarray:
    """
    Accept a file path (str), PIL Image, or numpy array.
    Returns a uint8 RGB array.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        if img.ndim == 2:                          # grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:                    # RGBA → RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img.astype(np.uint8)

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"), dtype=np.uint8)

    # file path
    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {source}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONTRAST ENHANCEMENT  (CLAHE on L-channel)
# ──────────────────────────────────────────────────────────────────────────────

def apply_clahe(rgb: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Operates in LAB color space so chroma is not distorted.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NOISE SUPPRESSION
# ──────────────────────────────────────────────────────────────────────────────

def denoise(rgb: np.ndarray,
            gaussian_ksize: int = 3,
            median_ksize: int = 3) -> np.ndarray:
    """
    Two-stage denoising:
      - Gaussian blur removes additive Gaussian noise
      - Median filter removes salt-and-pepper / impulse noise
    Both kernel sizes kept small to preserve tumour edges.
    """
    img = cv2.GaussianBlur(rgb, (gaussian_ksize, gaussian_ksize), sigmaX=0)
    img = cv2.medianBlur(img, median_ksize)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 4. SKULL STRIPPING / BRAIN EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_brain(rgb: np.ndarray,
                  threshold: int = 15,
                  morph_ksize: int = 7,
                  pad: int = 10) -> tuple[np.ndarray, tuple, np.ndarray]:
    """
    Remove the dark scanner background and crop to the brain region.

    Algorithm
    ---------
    1. Convert to grayscale.
    2. Binary threshold — pixels brighter than `threshold` belong to tissue.
    3. Morphological CLOSE fills small intra-cranial gaps.
    4. Morphological DILATE slightly expands the mask to catch skull edge.
    5. Find the largest external contour (the brain boundary).
    6. Create a convex-hull binary mask and apply it.
    7. Return the masked image and its tight bounding box.

    Returns
    -------
    cropped_rgb   : np.ndarray  — cropped, skull-stripped image (RGB)
    bbox          : tuple       — (x, y, w, h) in the original image coords
    brain_mask    : np.ndarray  — binary mask (same size as input rgb)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Close small holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Dilate to include skull periphery
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, k_dilate, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = rgb.shape[:2]
        return rgb.copy(), (0, 0, w, h), np.ones((h, w), dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)

    # Convex hull mask — removes any stray bright artifacts outside brain
    hull = cv2.convexHull(largest)
    brain_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(brain_mask, hull, 255)

    masked = cv2.bitwise_and(rgb, rgb, mask=brain_mask)

    # Bounding box with padding
    x, y, w, h = cv2.boundingRect(hull)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(rgb.shape[1] - x, w + 2 * pad)
    h = min(rgb.shape[0] - y, h + 2 * pad)

    cropped = masked[y:y+h, x:x+w]
    return cropped, (x, y, w, h), brain_mask


# ──────────────────────────────────────────────────────────────────────────────
# 5. TUMOR ROI SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def segment_tumor_roi(rgb: np.ndarray,
                      min_area_ratio: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised tumor region detection using multi-level Otsu thresholding
    and morphological refinement.

    This is NOT the DL classifier — it is an image-processing pre-analysis
    that highlights hyper-intense (bright) anomalous regions which are
    likely candidate tumor areas, to assist Grad-CAM validation.

    Algorithm
    ---------
    1. Grayscale + CLAHE for local contrast.
    2. Two-level Otsu (retval = single threshold) gives three intensity bands.
       We keep pixels in the top band (bright → potential tumour / enhancing tissue).
    3. Morphological OPEN removes salt noise; CLOSE fills holes.
    4. Remove tiny blobs (< min_area_ratio of image area).
    5. Return binary mask + coloured overlay.

    Returns
    -------
    seg_mask    : np.ndarray  — binary uint8 mask (255 = candidate region)
    overlay_img : np.ndarray  — RGB image with green contour overlay
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE to boost tumour-brain contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Two-level Otsu: pick upper-band threshold
    # cv2.threshold with THRESH_OTSU gives one threshold
    # For multi-level, apply twice on top half of histogram
    ret1, _ = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upper_mask = (gray_eq >= ret1).astype(np.uint8) * 255

    # Secondary Otsu on the bright pixels only
    bright_pixels = gray_eq[gray_eq >= ret1]
    if bright_pixels.size > 100:
        ret2, _ = cv2.threshold(bright_pixels, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Revert to the averaged heuristic: ret2 is too strict and destroys the tumor mask!
        final_thresh = int((ret1 + (ret1 + ret2) / 2) / 2)
    else:
        final_thresh = ret1

    _, seg = cv2.threshold(gray_eq, final_thresh, 255, cv2.THRESH_BINARY)

    # Morphological refinement
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN,  k3, iterations=2)
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, k7, iterations=3)

    # Remove small spurious regions and prevent whole-brain flooding
    min_area = min_area_ratio * rgb.shape[0] * rgb.shape[1]
    max_area = 0.45 * rgb.shape[0] * rgb.shape[1]  # tumors realistically don't span >45% of the brain flat
    
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(seg)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Coloured overlay
    overlay = rgb.copy()
    contours2, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours2, -1, (0, 255, 80), thickness=2)

    # Semi-transparent fill
    fill_layer = rgb.copy()
    cv2.drawContours(fill_layer, contours2, -1, (0, 255, 80), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 0.75, fill_layer, 0.25, 0)

    return clean_mask, overlay


# ──────────────────────────────────────────────────────────────────────────────
# 6. RESIZE FOR MODEL INPUT
# ──────────────────────────────────────────────────────────────────────────────

def resize_for_model(rgb: np.ndarray,
                     target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize to model input with aspect-ratio–preserving letterboxing,
    then pad with zeros to exactly `target_size`.
    Avoids distortion that simple cv2.resize introduces on non-square MRIs.
    """
    h, w = rgb.shape[:2]
    th, tw = target_size

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y_off = (th - nh) // 2
    x_off = (tw - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas


def normalize(rgb: np.ndarray) -> np.ndarray:
    """Float32 [0, 1] normalization."""
    return rgb.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# 7. AUGMENTATION  (training only)
# ──────────────────────────────────────────────────────────────────────────────

def random_augment(rgb: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply a random selection of augmentations suitable for medical MRI:
    - Horizontal / vertical flip
    - Random rotation ±20°
    - Random brightness / contrast jitter
    - Gaussian noise injection
    - Random zoom crop

    Does NOT apply colour jitter (MRI is grayscale information in RGB channels).
    """
    rng = np.random.default_rng(seed)
    img = rgb.copy()
    h, w = img.shape[:2]

    # Flip
    if rng.random() > 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if rng.random() > 0.7:
        img = cv2.flip(img, 0)  # vertical (less common but valid)

    # Rotation
    angle = rng.uniform(-20, 20)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    # Brightness & contrast
    alpha = rng.uniform(0.8, 1.2)   # contrast
    beta  = rng.uniform(-20, 20)    # brightness
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Zoom crop
    if rng.random() > 0.5:
        zoom = rng.uniform(0.85, 1.0)
        new_h, new_w = int(h * zoom), int(w * zoom)
        y0 = rng.integers(0, h - new_h) if h > new_h else 0
        x0 = rng.integers(0, w - new_w) if w > new_w else 0
        crop = img[y0:y0+new_h, x0:x0+new_w]
        img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # Gaussian noise
    if rng.random() > 0.6:
        sigma = rng.uniform(2, 8)
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 8. MASTER PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def full_pipeline(source,
                  target_size: tuple = (224, 224),
                  augment: bool = False,
                  return_intermediates: bool = False):
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    source             : path | PIL Image | np.ndarray
    target_size        : model input size (H, W)
    augment            : apply random augmentation (training mode)
    return_intermediates: if True, also return dict of stage outputs

    Returns
    -------
    model_input  : np.ndarray (1, H, W, 3) float32 — ready for model.predict()
    meta         : dict with intermediate images (only if return_intermediates)
    """
    raw          = load_image(source)
    enhanced     = apply_clahe(raw)
    denoised     = denoise(enhanced)
    cropped, bbox, brain_mask = extract_brain(denoised)
    seg_mask, seg_overlay     = segment_tumor_roi(cropped)
    resized      = resize_for_model(cropped, target_size)

    if augment:
        resized = random_augment(resized)

    normalized   = normalize(resized)
    model_input  = np.expand_dims(normalized, axis=0)

    if return_intermediates:
        meta = {
            "raw":          raw,
            "enhanced":     enhanced,
            "denoised":     denoised,
            "brain_mask":   brain_mask,
            "cropped":      cropped,
            "seg_mask":     seg_mask,
            "seg_overlay":  seg_overlay,
            "resized":      resized,
            "bbox":         bbox,
        }
        return model_input, meta

    return model_input


# ──────────────────────────────────────────────────────────────────────────────
# 9. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(source, save_path: str = None) -> plt.Figure:
    """
    Generate a diagnostic figure showing every stage of the pipeline.
    Useful for QA / paper figures.
    """
    _, meta = full_pipeline(source, return_intermediates=True)

    fig = plt.figure(figsize=(18, 10), facecolor="#0a0d12")
    fig.suptitle("NeuroScan AI — Image Processing Pipeline",
                 fontsize=15, color="#c8d0dc", fontweight="bold", y=0.98)

    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.15)

    panels = [
        (0, 0, "Raw Input",           meta["raw"],         "gray"),
        (0, 1, "CLAHE Enhanced",      meta["enhanced"],    "gray"),
        (0, 2, "Denoised",            meta["denoised"],    "gray"),
        (0, 3, "Brain Mask",          meta["brain_mask"],  "bone"),
        (1, 0, "Skull-Stripped Crop", meta["cropped"],     "gray"),
        (1, 1, "Segmented ROI",       meta["seg_overlay"], "gray"),
        (1, 2, "Tumor Mask (Otsu)",   meta["seg_mask"],    "hot"),
        (1, 3, "Model Input (224²)",  meta["resized"],     "gray"),
    ]

    for row, col, title, img, cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0a0d12")

        disp = img if img.ndim == 2 else img
        if img.ndim == 3 and cmap == "gray":
            # convert to grayscale for display consistency
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3:
            disp = img

        if disp.ndim == 2:
            ax.imshow(disp, cmap=cmap)
        else:
            ax.imshow(disp)

        ax.set_title(title, fontsize=9, color="#7a9fc0",
                     fontfamily="monospace", pad=5)
        ax.axis("off")

        # Draw bounding box on the raw image panel
        if title == "Raw Input" and meta.get("bbox"):
            x, y, w, h = meta["bbox"]
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor="#4da8ff", facecolor="none"
            )
            ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0d12")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(0)

    print(f"Running pipeline on: {path}")
    inp, meta = full_pipeline(path, return_intermediates=True)
    print(f"  model_input shape : {inp.shape}")
    print(f"  brain bbox        : {meta['bbox']}")
    print(f"  seg mask nonzero  : {np.count_nonzero(meta['seg_mask'])} px")

    fig = visualize_pipeline(path, save_path="pipeline_stages.png")
    print("  Saved pipeline_stages.png")
