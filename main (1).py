from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Tumor Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load Models
tumor_model = tf.keras.models.load_model("cnn_mode_binary.keras")  
type_model  = tf.keras.models.load_model("cnn_benign_malignant.h5")
unet_model  = tf.keras.models.load_model("unet_model.keras")
yolo_model  = YOLO("best.pt")


# ─────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array.reshape(1, 224, 224, 3)


def read_image_bytes(file: UploadFile) -> bytes:
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported.")
    return file.file.read()


# ─────────────────────────────────────────
# Model 1 — Tumor Detection
# ─────────────────────────────────────────
@app.post("/predict/tumor")
async def predict_tumor(file: UploadFile = File(...)):
    img = preprocess(read_image_bytes(file))

    try:
        pred = float(tumor_model.predict(img, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    label = "Tumor" if pred > 0.5 else "Normal"
    return {"prediction": label, "confidence": round(pred, 4)}


# ─────────────────────────────────────────
# Model 2 — Type
# ─────────────────────────────────────────
@app.post("/predict/type")
async def predict_type(file: UploadFile = File(...)):
    img = preprocess(read_image_bytes(file))

    try:
        pred = float(type_model.predict(img, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    label = "Malignant" if pred > 0.5 else "Benign"
    return {"prediction": label, "confidence": round(pred, 4)}


# ─────────────────────────────────────────
# Model 3 — YOLO
# ─────────────────────────────────────────
@app.post("/predict/detection")
async def predict_detection(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(read_image_bytes(file))).convert("RGB")
    img_array = np.array(image)

    try:
        results = yolo_model(img_array)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return {"boxes": [], "confidences": [], "count": 0}

        boxes = results[0].boxes.xyxy.tolist()
        confidences = [round(c, 4) for c in results[0].boxes.conf.tolist()]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"boxes": boxes, "confidences": confidences, "count": len(boxes)}


# ─────────────────────────────────────────
# Model 4 — U-Net
# ─────────────────────────────────────────
@app.post("/predict/segmentation")
async def predict_segmentation(file: UploadFile = File(...)):
    img = preprocess(read_image_bytes(file))

    try:
        mask = unet_model.predict(img, verbose=0)[0][:, :, 0]
        binary_mask = (mask > 0.5).astype(float)
        coverage = round(float(binary_mask.mean()) * 100, 2)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "coverage_percent": coverage
    }


# ─────────────────────────────────────────
# Full Pipeline 
# ─────────────────────────────────────────
@app.post("/predict/full")
async def full_pipeline(file: UploadFile = File(...)):
    img_bytes = read_image_bytes(file)
    img = preprocess(img_bytes)

    try:
        # Step 1: Tumor detection
        tumor_pred = float(tumor_model.predict(img, verbose=0)[0][0])

        if tumor_pred <= 0.5:
            return {
                "has_tumor": False,
                "prediction": "Normal"
            }

        # Step 2: Type
        type_pred = float(type_model.predict(img, verbose=0)[0][0])
        tumor_type = "Malignant" if type_pred > 0.5 else "Benign"

        # Step 3: YOLO
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        yolo_results = yolo_model(np.array(image))

        if yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
            boxes = []
        else:
            boxes = yolo_results[0].boxes.xyxy.tolist()

        # Step 4: U-Net
        mask = unet_model.predict(img, verbose=0)[0][:, :, 0]
        coverage = round(float((mask > 0.5).mean()) * 100, 2)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "has_tumor": True,
        "type": tumor_type,
        "confidence": round(type_pred, 4),
        "boxes": boxes,
        "coverage_percent": coverage
    }


# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": 4}


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)