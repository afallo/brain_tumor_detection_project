from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import io
import uvicorn
import numpy as np
from PIL import Image
from brain.params import *
from brain.registry import load_model, load_model_seg2D, load_model_docker
from brain.ml_logic_classification.preprocess import preprocess_for_inference
from brain.ml_logic_segmentation_2D.preprocess import preprocess_loaded_image_for_inference
from brain.ml_logic_segmentation_2D.metrics import dice_coef, dice_coef_loss

app = FastAPI()





@app.get("/")
def read_root():
    return {"message": "Hello brainers"}

@app.post("/predict_classification")
async def predict_class(file: UploadFile = File(...)):
    # Lire l'image
    model = load_model_docker("models/classification.h5")

    img = Image.open(file.file).convert("RGB")

    img_array = preprocess_for_inference(img)
      # Prédiction

    preds = model.predict(img_array)

    predicted_class = int(np.argmax(preds))

    return {"class": predicted_class, "scores": preds.tolist()}



@app.post("/predict_segmentation_2D")
async def predict_seg2D(file: UploadFile = File(...)):
    # Lire l'image

    model = load_model_docker("models/seg2D.h5")

    img = Image.open(file.file).convert("RGB")
    img_np = np.array(img)

    img_pre = preprocess_loaded_image_for_inference(img_np, IMG_SIZE)
    print("shape", img_pre.shape)
      # Prédiction


    pred_mask = model.predict(img_pre)

    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    # On retire la dimension du batch pour l'affichage (256, 256)
    pred_mask = np.squeeze(pred_mask)

    # Convertir le masque en image PIL
    mask_img = Image.fromarray(pred_mask.squeeze() * 255)  # Assure-toi que le masque est en 0-255
    # Sauvegarder dans un buffer
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    buffer.seek(0)
    # Retourner le buffer comme réponse
    return Response(content=buffer.getvalue(), media_type="image/png")
