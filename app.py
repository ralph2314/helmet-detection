from flask import Flask, render_template, request
from PIL import Image
import os
import torch

app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)

model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        torch.set_num_threads(1)
        model = YOLO("helmet_best.pt")
        model.to("cpu")
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join("static/uploads", "detected.jpg")
    img = Image.open(file).convert("RGB")
    img = img.resize((320, 320))
    img.save(filepath)
    m = get_model()
    results = m(filepath, imgsz=320)
    results[0].save(filename=filepath)
    return render_template("index.html", image="static/uploads/detected.jpg")

if __name__ == "__main__":
    app.run()
