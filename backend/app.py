from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
MODEL_SERVICE = os.getenv("MODEL_SERVICE_URL", "http://model-service:8001")
print(f"Model Service URL: {MODEL_SERVICE}")  # Debug on startup

@app.get("/")
def home():
    return {"status": "Backend API running"}

@app.get("/models")
def get_models():
    """Get available models"""
    try:
        client = mlflow.tracking.MlflowClient()
        models = []

        print(client.search_registered_models())

        for rm in client.search_registered_models():
            for version in rm.latest_versions:
                if version.current_stage == "Production":
                    models.append({
                        "name": rm.name,
                        "version": version.version,
                        "stage": version.current_stage
                    })
        
        # Default if no models
        if not models:
            raise Exception("No Registered Models were found.")
        
        return {"models": models}
    except Exception as e:
        return {"Error": e}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    """Forward prediction request to model service"""
    try:
        # Read file
        content = await file.read()
        print("File name: ", file.filename)

        # Passing arguments to model service
        files = {"file": (file.filename, content, file.filename.split('.')[-1])}
        data = {"model_name": model_name}
        
        # Make request to model service
        print("📤 Sending request to model service...")
        response = requests.post(
            f"{MODEL_SERVICE}/predict",
            files=files,
            data=data,
            timeout=120  # Increased timeout for video processing
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error response: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Model service error: {response.text}"
            )
        
        result = response.json()
        print(f"✅ Success: {len(result.get('detections', []))} detections")
        return result
    
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Cannot connect to model service at {MODEL_SERVICE}"
        )
    
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        raise HTTPException(
            status_code=504, 
            detail="Model service request timed out"
        )
    
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )