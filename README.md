# Breast Cancer Detection System

An AI-powered Breast Cancer Detection System using YOLOv8 and FastAPI for tumor detection, localization, and classification from mammogram images.

---

## Overview

This project uses Deep Learning and Computer Vision techniques to detect breast cancer from mammography images.

The system performs:

* Tumor Detection
* Tumor Localization using Bounding Boxes
* Classification (Cancer / Normal)
* Real-time Prediction through FastAPI API

The model was trained using YOLOv8 on annotated medical images.

---

## Dataset

The dataset is organized in YOLO format.

Each image has a corresponding `.txt` label file with the same filename.

Label format:

```txt
class x_center y_center width height
```

Example:

```txt
0 0.41 0.39 0.11 0.15
```

Where:

* `0` → Cancer
* `1` → Normal

The remaining values represent the bounding box coordinates normalized between 0 and 1.

---

## Classes

| Class ID | Label  |
| -------- | ------ |
| 0        | Cancer |
| 1        | Normal |

---

## YOLO Data Configuration

The project uses a `data.yaml` file to define:

* Dataset paths
* Train / Validation / Test directories
* Number of classes
* Class names

Example:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['cancer', 'normal']
```

---

## Model

This project uses:

* YOLOv8
* Ultralytics Framework
* Python
* FastAPI

The model performs:

* Object Detection
* Classification
* Localization

at the same time.

---

## Training

The model was trained using:

* Training images → for learning
* Validation images → for evaluation after each epoch
* Test images → for final testing

Training was performed for multiple epochs using GPU acceleration.

---

## Evaluation Metrics

Since YOLO performs Detection + Classification together, the evaluation is based on:

* mAP@50
* mAP@50:95

Model performance reached approximately:

* 91% – 92% mAP

---

## API Integration

A FastAPI backend was developed for real-time predictions.

The API:

* Receives an image
* Runs YOLO inference
* Returns:

  * Prediction
  * Confidence Score
  * Output image with Bounding Box

---

## API Endpoints

### Home Endpoint

```http
GET /
```

### Prediction Endpoint

```http
POST /predict
```

---

## Example API Response

```json
{
  "prediction": "cancer",
  "confidence": 85.37,
  "image": "http://127.0.0.1:8000/image/result.jpg"
}
```

---

## Technologies Used

* Python
* YOLOv8
* Ultralytics
* OpenCV
* FastAPI
* Uvicorn

---

## Project Structure

```bash
project/
│
├── main.py
├── best.pt
├── data.yaml
├── requirements.txt
├── uploads/
├── results/
└── README.md
```

---

## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload
```

Open Swagger Docs:

```bash
http://127.0.0.1:8000/docs
```

---

## Output Example

The model returns the uploaded mammogram image with detected tumor regions highlighted using bounding boxes.

---

## Future Improvements

* Deploy the API publicly
* Improve dataset size
* Add segmentation support
* Build full web integration
* Support DICOM medical images

---

## Author

Mohamed Karim
