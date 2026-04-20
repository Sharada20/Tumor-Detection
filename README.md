# 🧠 NeuroScan AI — Brain Tumor Diagnostic Assistant

A Streamlit web application that classifies brain MRI scans into one of four categories using deep learning, and explains its prediction via **Grad-CAM heatmap overlays**.

> ⚠️ **Research use only.** This tool is not a certified medical device and must not replace qualified clinical judgment.

---

## Features

| Feature | Details |
|---|---|
| **Classification** | Glioma · Meningioma · Pituitary · No Tumor |
| **Auto-cropping** | Morphological preprocessing removes dark background before inference |
| **Explainability** | Grad-CAM highlights the exact CNN region that drove the prediction |
| **Confidence display** | Per-class probability distribution shown as bars |
| **Download** | Heatmap overlay exportable as PNG |

---

## Project Structure

```
brain_tumor_app/
├── app.py               # Main Streamlit application
├── style.css            # Custom dark clinical UI styles
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml      # Theme & server configuration
└── README.md
```

---

## Quick Start

### 1. Clone / download this folder

```bash
cd brain_tumor_app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Google Drive model file ID

Open `app.py` and replace `YOUR_GDRIVE_FILE_ID` with the actual file ID from your Google Drive share link.

**How to get the file ID:**
- Share your `.h5` model file on Google Drive (set to "Anyone with the link")
- Copy the link — it looks like:  
  `https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view`
- The file ID is the long alphanumeric string:  
  `1AbCdEfGhIjKlMnOpQrStUvWxYz`

```python
# app.py — line ~40
file_id = "1AbCdEfGhIjKlMnOpQrStUvWxYz"   # ← paste your ID here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Model Requirements

The app expects a **Keras/TensorFlow `.h5` model** with:

| Property | Value |
|---|---|
| Input shape | `(None, 224, 224, 3)` |
| Output | Softmax over 4 classes |
| Class order | `[Glioma, Meningioma, No Tumor, Pituitary]` |
| Architecture | Any CNN with at least one `Conv2D` layer (required for Grad-CAM) |

### Recommended architecture

A fine-tuned **EfficientNetB0** or **MobileNetV2** on the  
[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) works well.

**Example training snippet:**
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = True   # fine-tune all layers

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(4, activation="softmax"),   # 4 classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.save("brain_tumor_model.h5")
```

---

## How Grad-CAM Works

1. A **gradient model** is built that exposes the last `Conv2D` layer's output alongside the final predictions.  
2. `GradientTape` records the gradient of the **target class score** w.r.t. the conv feature maps.  
3. Gradients are **global-average-pooled** to get per-channel importance weights.  
4. A weighted sum of feature maps produces a raw heatmap, which is ReLU'd and normalised.  
5. The heatmap is **bilinearly upsampled** to the original image size and blended with `cv2.applyColorMap(COLORMAP_JET)`.

Red/yellow areas = high activation → most influential for the prediction.  
Blue/green areas = low activation → largely ignored by the model.

---

## Preprocessing Pipeline

```
Uploaded image
     │
     ▼
Convert to RGB
     │
     ▼
Grayscale → threshold (>15) → morphological CLOSE
     │
     ▼
Find largest external contour  ←── background removal
     │
     ▼
Bounding-box crop + 10px padding
     │
     ▼
Resize to 224×224
     │
     ▼
Normalize to [0, 1]
     │
     ▼
Expand dims → model input batch
```

---

## Deployment

### Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set `app.py` as the entry point.
4. The model is downloaded from Google Drive on first run and cached.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t neuroscan-ai .
docker run -p 8501:8501 neuroscan-ai
```

---

## Disclaimer

This software is provided for **research and educational purposes only**.  
It has not been validated as a medical device and must not be used for clinical diagnosis.  
Always consult a qualified radiologist or neurosurgeon for medical decisions.
