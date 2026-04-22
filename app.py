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

def apply_nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

def draw_boxes(image, outputs, threshold=0.65):
    img = np.array(image)
    h, w = img.shape[:2]
    predictions = outputs[0][0]
    boxes = []
    scores = []
    class_ids = []
    for pred in predictions.T:
        x, y, bw, bh = pred[:4]
        pscores = pred[4:]
        score = float(np.max(pscores))
        class_id = int(np.argmax(pscores))
        if score < threshold:
            continue
        x1 = max(0, int((x - bw/2)))
        y1 = max(0, int((y - bh/2)))
        x2 = min(w, int((x + bw/2)))
        y2 = min(h, int((y + bh/2)))
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    summary = {}
    if len(boxes) > 0:
        keep = apply_nms(np.array(boxes), np.array(scores))
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            label_name = CLASSES[class_id] if class_id < len(CLASSES) else "object"
            color = COLORS.get(label_name, (0, 255, 0))
            label = f"{label_name} {score*100:.0f}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
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
