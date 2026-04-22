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

CLASSES = ["head", "helmet", "hi-viz helmet", "hi-viz vest", "person", "random"]
COLORS = {
    "helmet": (0, 200, 0),
    "hi-viz helmet": (0, 180, 0),
    "head": (0, 0, 255),
    "person": (255, 140, 0),
    "hi-viz vest": (200, 200, 0),
    "random": (128, 128, 128)
}

def preprocess(img):
    img = img.resize((320, 320))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def draw_boxes(image, outputs, threshold=0.5):
    img = np.array(image)
    h, w = img.shape[:2]
    predictions = outputs[0][0]
    summary = {}
    for pred in predictions.T:
        x, y, bw, bh = pred[:4]
        scores = pred[4:]
        score = float(np.max(scores))
        class_id = int(np.argmax(scores))
        if score < threshold:
            continue
        x1 = max(0, int((x - bw/2)))
        y1 = max(0, int((y - bh/2)))
        x2 = min(w, int((x + bw/2)))
        y2 = min(h, int((y + bh/2)))
        label_name = CLASSES[class_id] if class_id < len(CLASSES) else "object"
        color = COLORS.get(label_name, (0, 255, 0))
        label = f"{label_name} {score*100:.0f}%"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
        cv2.putText(img, label, (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if label_name not in summary:
            summary[label_name] = []
        summary[label_name].append(round(score*100, 1))
    return Image.fromarray(img), summary

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    inp = preprocess(img)
    outputs = session.run(None, {input_name: inp})
    result, summary = draw_boxes(img.resize((320, 320)), outputs)
    filepath = "static/uploads/detected.jpg"
    result.save(filepath)
    return render_template("index.html", image=filepath, summary=summary)

if __name__ == "__main__":
    app.run()
