from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import cv2
import uuid
import os

app = FastAPI(
    title="Breast Cancer Detection API"
)

# Load trained model
model = YOLO("best.pt")

UPLOAD_FOLDER="uploads"

os.makedirs(UPLOAD_FOLDER,exist_ok=True)


@app.get("/")
def home():
    return {"message":"API Working"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_path=os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(image_path,"wb") as f:
        f.write(await file.read())

    results=model(image_path)

    plotted=results[0].plot()

    result_name=f"result_{uuid.uuid4()}.jpg"

    result_path=os.path.join(
        UPLOAD_FOLDER,
        result_name
    )

    cv2.imwrite(result_path,plotted)

    boxes=results[0].boxes

    if len(boxes)>0:

        cls=int(boxes[0].cls)
        conf=float(boxes[0].conf)

        prediction=model.names[cls]

        return JSONResponse(
            content={
                "prediction":prediction,
                "confidence":round(conf*100,2),
                "image":f"http://127.0.0.1:8000/image/{result_name}"
            }
        )

    return JSONResponse(
        content={
            "prediction":"No Detection",
            "confidence":0,
            "image":f"http://127.0.0.1:8000/image/{result_name}"
        }
    )


from fastapi.responses import FileResponse

@app.get("/image/{image_name}")
async def get_image(image_name:str):

    path=os.path.join(
        UPLOAD_FOLDER,
        image_name
    )

    return FileResponse(path)