from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/2.keras")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get('/ping')
async def ping():
    return "Hello there, hi"

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # we need to chnge this contents in to numpy array
    image = read_file_as_image(contents)
    image_batch = np.expand_dims(image,0)
    index = np.argmax(MODEL.predict(image_batch))
    confidense = 100*np.max(MODEL.predict(image_batch))
    pred_class = CLASS_NAME[index]
    print(index)
    return {
        "Class" : pred_class,
        "Confidence": confidense
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)