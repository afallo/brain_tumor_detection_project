from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
from brain.params import *
from brain.registry import load_model
from brain.ml_logic_classification.preprocess import preprocess_for_inference



app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image
    model = load_model()
    print("j'ai le model")
    img = Image.open(file.file).convert("RGB")

    print("j'ai récupéré l'image")

    img_array = preprocess_for_inference(img)
    print("j'ai preprocess en array l'image")


    # Prédiction
    preds = model.predict(img_array)
    print("j'ai le pred")
    predicted_class = int(np.argmax(preds))

    return {"class": predicted_class, "scores": preds.tolist()}

@app.get("/")
def read_root():
    return {"message": "Hello brainers"}
