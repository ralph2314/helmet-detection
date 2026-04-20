from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

model = YOLO("helmet_best.pt")

os.makedirs("static/uploads", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join("static/uploads", "detected.jpg")
    img = Image.open(file).convert("RGB")
    img.save(filepath)
    results = model(filepath)
    results[0].save(filename=filepath)
    return render_template("index.html", image="static/uploads/detected.jpg")

if __name__ == "__main__":
    app.run()