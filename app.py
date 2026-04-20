import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

from preprocessing import full_pipeline

# --- Page Configuration (must be first Streamlit call) ---
st.set_page_config(
    page_title="NeuroScan AI · Tumor Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Load custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    model = efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, 4)
    )
    model_path = "brain_tumor_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    else:
        st.error("No brain_tumor_model.pth found! Please run train.py first.")
        st.stop()
    model.eval()
    return model

def preprocess_image(pil_image: Image.Image, target_size=(224, 224)):
    model_input, meta = full_pipeline(pil_image, target_size=target_size, augment=False, return_intermediates=True)
    
    # model_input shape is (1, H, W, 3) float32, we need (1, 3, H, W)
    img_tensor = torch.from_numpy(model_input.transpose((0, 3, 1, 2)))
    
    return img_tensor, Image.fromarray(meta["cropped"]), Image.fromarray(meta["seg_overlay"])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        preds = self.model(x)
        self.model.zero_grad()
        loss = preds[0, class_idx]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, (x.shape[3], x.shape[2]))
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap, torch.softmax(preds, dim=1).detach().cpu().numpy()[0]

def overlay_heatmap(original_pil: Image.Image, heatmap: np.ndarray, alpha=0.45, draw_bbox=False):
    orig_array = np.array(original_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig_array.shape[1], orig_array.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_array, 1 - alpha, colored_rgb, alpha, 0)
    
    if draw_bbox:
        binary_mask = np.uint8(heatmap_resized > 0.50) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

    return Image.fromarray(overlay)

# --- Constants ---
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = {
    "Glioma":     "#FF4B4B",
    "Meningioma": "#FF9800",
    "No Tumor":   "#4CAF50",
    "Pituitary":  "#2196F3",
}
CLASS_DESC = {
    "Glioma":     "A tumor arising from glial cells. Can be low- or high-grade (e.g. GBM). Requires immediate specialist review.",
    "Meningioma": "Typically benign tumor of the meninges. Often slow-growing; treatment depends on size and location.",
    "No Tumor":   "No evidence of a tumor mass detected in this scan. Recommend clinical correlation.",
    "Pituitary":  "Tumor of the pituitary gland. Usually benign adenoma; hormonal assessment advised.",
}

# ============================================================
# UI
# ============================================================

st.markdown('''
<div class="header-wrap">
  <div class="brand">
    <span class="brand-icon">🧠</span>
    <div>
      <span class="brand-name">NeuroScan</span>
      <span class="brand-tagline">Brain Tumor Detection</span>
    </div>
<div class="divider"></div>
''', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.35], gap="large")

with col_upload:
    st.markdown('<p class="section-label">01 — Upload MRI Scan</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-hint">Accepted formats: JPG · PNG · BMP · TIFF</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "tiff"], label_visibility="collapsed")

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<p class="img-label">Original Upload</p>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)

        st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
        run_analysis = st.button("Run Diagnostic Analysis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="upload-placeholder">
          <div class="upload-icon">📂</div>
          <p>Drag & drop an MRI image here<br>or click <strong>Browse files</strong> above</p>
        </div>
        ''', unsafe_allow_html=True)
        run_analysis = False

with col_result:
    st.markdown('<p class="section-label">02 — Analysis Results</p>', unsafe_allow_html=True)

    if uploaded_file and run_analysis:
        with st.spinner("🔬 Analysing scan…"):
            try:
                model = load_model()
                img_batch, cropped_pil, seg_overlay_pil = preprocess_image(pil_image)

                target_layer = model.features[-1]
                grad_cam = GradCAM(model, target_layer)
                
                preds = model(img_batch)
                class_idx = int(torch.argmax(preds, dim=1))
                label = CLASS_LABELS[class_idx]
                
                heatmap, probs = grad_cam(img_batch, class_idx)
                confidence = float(probs[class_idx]) * 100
                color = CLASS_COLORS[label]
                
                has_tumor = (label != "No Tumor")
                overlay_img = overlay_heatmap(pil_image, heatmap, draw_bbox=has_tumor)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # classification card
        st.markdown(f'''
        <div class="result-card" style="border-color:{color};">
          <div class="result-header">
            <span class="result-label">Classification</span>
            <span class="result-badge" style="background:{color};">{label}</span>
          </div>
          <div class="confidence-row">
            <span class="conf-value">{confidence:.1f}%</span>
            <span class="conf-text">model confidence</span>
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fg" style="width:{confidence:.1f}%; background:{color};"></div>
          </div>
          <p class="result-desc">{CLASS_DESC[label]}</p>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:1.4rem;">Probability Distribution</p>', unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(zip(CLASS_LABELS, probs)):
            pct = prob * 100
            is_top = i == class_idx
            cls_color = CLASS_COLORS[cls]
            st.markdown(f'''
            <div class="prob-row {'prob-top' if is_top else ''}">
              <span class="prob-name">{cls}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fg" style="width:{pct:.1f}%; background:{cls_color};"></div>
              </div>
              <span class="prob-pct" style="color:{cls_color};">{pct:.1f}%</span>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:1.6rem;">03 — Explainability · Grad-CAM Heatmap</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-hint">Red/yellow regions are the areas the model weighted most heavily for its decision.</p>', unsafe_allow_html=True)

        img_col1, img_col2 = st.columns(2, gap="small")
        with img_col1:
            st.markdown('<p class="img-label">Skull-Stripped Crop</p>', unsafe_allow_html=True)
            st.image(cropped_pil, use_container_width=True)
        with img_col2:
            st.markdown('<p class="img-label">Grad-CAM Overlay</p>', unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)

        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Heatmap",
            data=buf.getvalue(),
            file_name=f"gradcam_{label.lower().replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )


    elif not uploaded_file:
        st.markdown('''
        <div class="empty-state">
          <div class="empty-icon">🔬</div>
          <p>Upload a brain MRI scan on the left to begin the analysis.</p>
          <ul class="feature-list">
            <li>✦ Automatic brain-region cropping</li>
            <li>✦ Deep learning classification (4 classes)</li>
            <li>✦ Grad-CAM explainability heatmap</li>
            <li>✦ Downloadable overlay image</li>
          </ul>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="empty-state">
          <div class="empty-icon">👈</div>
          <p>Click <strong>Run Diagnostic Analysis</strong> to start.</p>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('''
<div class="footer">
  NeuroScan AI · Built with PyTorch & Streamlit · 

</div>
''', unsafe_allow_html=True)
