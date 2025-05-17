import torch
import sys
import os
import mlflow
import mlflow.pytorch
import requests
import json
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="COVID-19 Classifier")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# MLflow model serving endpoint
MLFLOW_SERVING_URL = "http://localhost:1234/invocations"

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Convert to numpy array and prepare for MLflow serving
        input_data = {
            "inputs": image_tensor.numpy().tolist()
        }
        
        # Send request to MLflow serving endpoint
        response = requests.post(
            MLFLOW_SERVING_URL,
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Model serving error")
            
        # Parse response
        predictions = response.json()
        probabilities = np.array(predictions["predictions"][0])
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        # Get class names
        class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        class_name = class_names[predicted_class]
        
        # Get all class probabilities
        class_probabilities = {
            class_names[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "prediction": class_name,
            "confidence": confidence,
            "all_probabilities": class_probabilities
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 