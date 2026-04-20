import os
import cv2
import argparse
from pathlib import Path
from preprocessing import full_pipeline
import multiprocessing

def process_image(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return
    try:
        # We use return_intermediates=True to get the final uint8 resized image before float normalization
        _, meta = full_pipeline(str(src_path), target_size=(224, 224), augment=False, return_intermediates=True)
        img = meta["resized"]
        
        # Convert RGB back to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img_bgr)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=".", help="Directory containing Training and Testing folders")
    parser.add_argument("--dest", type=str, default="data_preproc", help="Destination directory for processed dataset")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    tasks = []
    
    # Process both Training and Testing folders
    for split in ["Training", "Testing"]:
        split_dir = source_dir / split
        if not split_dir.exists():
            print(f"Warning: Could not find {split_dir}")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    # Create corresponding destination path
                    rel_path = img_path.relative_to(source_dir)
                    dst_path = str(dest_dir / rel_path)
                    
                    # Convert extension to .jpg for uniformity
                    dst_path = os.path.splitext(dst_path)[0] + ".jpg"
                    tasks.append((str(img_path), dst_path))

    total = len(tasks)
    print(f"Found {total} images to process. Starting batch preprocessing...")
    
    # Run in parallel using all CPU cores
    if tasks:
        with multiprocessing.Pool() as pool:
            for i, _ in enumerate(pool.imap_unordered(process_image, tasks), 1):
                if i % 100 == 0 or i == total:
                    print(f"Progress: [{i}/{total}] images processed")
                    
        print(f"\nOptimization complete! Processed dataset saved to: '{dest_dir.absolute()}'")
    else:
        print("No images found to process.")

if __name__ == "__main__":
    main()
