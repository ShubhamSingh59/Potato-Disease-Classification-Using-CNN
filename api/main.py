from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

PORT = 8000

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/2.keras")
CLASS_NAME = ['Early_blight', 'Late_blight', 'Healthy']

@app.get('/ping')
async def ping():
    return "Hello there, hi"

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_file_as_image(contents)
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    confidence = float(np.max(predictions))  
    class_index = np.argmax(predictions)
    return {
        "class": CLASS_NAME[class_index],  
        "confidence": confidence  
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=PORT)