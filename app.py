from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import onnxruntime as rt
import cv2
import os

app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)

session = rt.InferenceSession("helmet_best.onnx")
input_name = session.get_inputs()[0].name

def preprocess(img):
    img = img.resize((320, 320))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def draw_boxes(image, outputs, threshold=0.5):
    img = np.array(image)
    predictions = outputs[0][0]
    for pred in predictions.T:
        x, y, w, h = pred[:4]
        scores = pred[4:]
        score = float(np.max(scores))
        if score < threshold:
            continue
        x1 = int((x - w/2))
        y1 = int((y - h/2))
        x2 = int((x + w/2))
        y2 = int((y + h/2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return Image.fromarray(img)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    inp = preprocess(img)
    outputs = session.run(None, {input_name: inp})
    result = draw_boxes(img.resize((320,320)), outputs)
    filepath = "static/uploads/detected.jpg"
    result.save(filepath)
    return render_template("index.html", image=filepath)

if __name__ == "__main__":
    app.run()
