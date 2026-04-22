Breast Cancer Detection System

An end-to-end AI system for detecting, classifying, and segmenting breast tumors using deep learning models and FastAPI.

⸻

Overview

This project integrates multiple AI models to analyze medical images and provide:
	•	Tumor detection (Tumor / Normal)
	•	Tumor classification (Benign / Malignant)
	•	Object detection using YOLO
	•	Image segmentation using U-Net

The system is designed for backend integration using FastAPI.

⸻

Models

1. CNN - Tumor Detection
	•	Input: 224x224 RGB image
	•	Output: Binary (0 = Normal, 1 = Tumor)
	•	Activation: Sigmoid

2. CNN - Tumor Type Classification
	•	Input: 224x224 RGB image
	•	Output: Binary (0 = Benign, 1 = Malignant)
	•	Activation: Sigmoid

3. YOLOv8 - Object Detection
	•	Input: Original image
	•	Output: Bounding boxes with confidence scores
	•	Model file: best.pt

4. U-Net - Segmentation
	•	Input: 224x224 RGB image
	•	Output: Segmentation mask (224x224)
	•	Threshold: 0.5

⸻

Preprocessing
	•	Image resized to 224x224
	•	Converted to RGB
	•	Normalized by dividing by 255.0
	•	Data type: float32

⸻

API Endpoints

POST /predict/tumor

Detect if image contains a tumor
Response  
{
  "prediction": "Benign or Malignant",
  "confidence": 0.93
}


POST /predict/type

Classify tumor as benign or malignant
Response:
{
  "prediction": "Benign or Malignant",
  "confidence": 0.93
}
POST /predict/detection

Detect tumor bounding boxes
Response:
{
  "boxes": [[x1, y1, x2, y2]],
  "confidences": [0.91],
  "count": 1
}
POST /predict/segmentation

Generate tumor mask
Response: {
  "mask": [[0,1,1,...]],
  "coverage_percent": 12.5
}
project/
│
├── main.py
├── requirements.txt
├── models_links.txt
├── README.md
│
├── notebooks/
│   ├── cnn_binary.ipynb
│   ├── cnn_type.ipynb
│   ├── yolo.ipynb
│   └── unet.ipynb
