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
        
        response = requests.post(
            f"http://localhost:8001/predict",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )