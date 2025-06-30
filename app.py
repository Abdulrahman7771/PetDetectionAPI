"""
app.py
Flask entry point exposing /analyze for your ASP.NET MVC front‑end.
"""

from flask import Flask, request, jsonify
from model_utils import read_image_from_base64, classify, detect, segment

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    img_b64 = data.get("image")

    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    img = read_image_from_base64(img_b64)

    # 1️⃣ Classification quick filter
    if not classify(img):
        return jsonify({"found": False})

    # 2️⃣ Detection and segmentation (loaded one‑by‑one inside helpers)
    boxes = detect(img)
    mask = segment(img)

    return jsonify({"found": True, "boxes": boxes, "mask": mask})


if __name__ == "__main__":
    app.run(debug=True)
