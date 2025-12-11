from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

# Checking Resulting mlflow configuration
print(os.getenv("MLFLOW_TRACKING_URI"))
print(os.getenv("MLFLOW_TRACKING_USERNAME"))
print(os.getenv("MLFLOW_TRACKING_PASSWORD"))

app = FastAPI()

# Cache loaded models
models = {}

def load_model(model_name: str):
    """Load model from MLflow or use default"""
    if model_name in models:
        return models[model_name]
    
    print(f"Loading model: {model_name}")
    
    # Try MLflow first
    try:
        # Making mlflow client
        client = MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        print("Model Version:", model_version)

        # Get the run ID associated with this model version
        run_id = model_version.run_id
        print("Run ID: ", run_id)

        # Making local direcotry if doesn't exist
        local_path = f"models/{model_name}"
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)

        # Downloading artifacts
        downloaded_path = client.download_artifacts(
            run_id=run_id,
            path="weights",  # This downloads only the weights subdirectory
            dst_path=local_path
        )
        print("Downloaded path: ", downloaded_path)
        
        # Find .pt file in the downloaded weights directory
        pt_files = list(Path(local_path).rglob("*.pt"))
        if pt_files:
            for pt in pt_files:
                if pt.name == "best.pt":
                    print("Model saved: ", str(pt))
                    model = YOLO(str(pt))
                    models[model_name] = model
                    return model
        else:
            raise Exception("No .pt file found in weights directory")
    except Exception as e:
        print(f"MLflow model load failed: {e}")

@app.get("/")
def home():
    return {"status": "Model service running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        start = time.time()

        content = await file.read()
        model = load_model(model_name)
        
        ext = file.filename.split('.')[-1].lower()
        if ext in {"jpg", "jpeg", "png"}:
            detections = process_image(content, model)
        elif ext == "mp4":
            detections = process_video(content, model)
        else:
            raise HTTPException(400, f"Unsupported file format: {file.filename}")
            
        return {
            "success": True,
            "model_used": model_name,
            "processing_time": time.time() - start,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_image(image_bytes, model):
    """Process single image"""
    # Convert to array
    print("Process image called")

    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(img_array, conf=0.25)
    
    # Extract detections
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    
    return detections

def process_video(video_bytes, model):
    """Process video using YOLO's built-in video handler"""
    print("Process video called")

    temp_path = "temp/video.mp4"
    with open(temp_path, 'wb') as f:
        f.write(video_bytes)
    
    # Let YOLO handle the video processing
    results = model(temp_path, conf=0.25, stream=True)
    
    detections = []
    for frame_idx, result in enumerate(results):
        for box in result.boxes:
            detections.append({
                "frame": frame_idx,
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    
    os.remove(temp_path)
    return detections

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8001,
        reload=True
    )