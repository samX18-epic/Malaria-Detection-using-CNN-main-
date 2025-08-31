from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-lovable-app-domain.com"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (change the path accordingly)
MODEL_PATH = "malaria_detection by sam (2).keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    # Open image, convert to RGB, resize, normalize, and reshape for model input
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))  # Replace with your model's expected size
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 64, 64, 3)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = model.predict(image)[0][0]
        label = "Parasitized" if prediction > 0.5 else "Uninfected"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Malaria Detection API is live. Use POST /predict with an image file."}
