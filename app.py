from flask import Flask, request, jsonify
from model_utils import read_image_from_base64, classify, detect, segment

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    image = read_image_from_base64(image_b64)

    if not classify(image):
        return jsonify({ "found": False })

    boxes = detect(image)
    masks = segment(image)

    return jsonify({
        "found": True,
        "boxes": boxes,
        "mask": masks
    })

if __name__ == "__main__":
    app.run(debug=True)
