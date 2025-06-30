import base64
import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, segmentation
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────────────────────────
# Constants ‑ labels the same every call (tiny, so we keep them global)
# ────────────────────────────────────────────────────────────────────────────────
IMAGENET_LABELS = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

VOC_LABELS = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ────────────────────────────────────────────────────────────────────────────────
# Helper: convert base‑64 string → PIL.Image (RGB)
# ────────────────────────────────────────────────────────────────────────────────
def read_image_from_base64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    img_np = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ────────────────────────────────────────────────────────────────────────────────
# 1. Classification  ‑ returns True if cat/dog keyword appears
# ────────────────────────────────────────────────────────────────────────────────
def classify(img: Image.Image) -> bool:
    # 🔸 Load model *inside* the function
    model = resnet50(weights="IMAGENET1K_V1")
    model.eval()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
    pred_id = logits.argmax(1).item()
    label = IMAGENET_LABELS[pred_id].lower()

    # 🔸 Free memory
    del model, logits
    torch.cuda.empty_cache()

    return ("cat" in label) or ("dog" in label)


# ────────────────────────────────────────────────────────────────────────────────
# 2. Detection  ‑ returns a list of [x1, y1, x2, y2] boxes for cats & dogs only
# ────────────────────────────────────────────────────────────────────────────────
def detect(image: Image.Image):
    yolo_model = YOLO("yolov5su.pt")
    results = yolo_model.predict(image, conf=0.5)[0]
    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        label = yolo_model.model.names[cls]
        if label.lower() in ["cat", "dog"]:
            coords = box.xyxy[0].cpu().numpy().tolist()
            coords = [float(c) for c in coords]  # force float conversion
            boxes.append(coords)
    return boxes  # List[List[float]]


# ────────────────────────────────────────────────────────────────────────────────
# 3. Segmentation  ‑ returns binary mask (0=background, 1=pet pixels)
# ────────────────────────────────────────────────────────────────────────────────\

def segment(image: Image.Image):
    seg_model = segmentation.deeplabv3_resnet50(weights="DEFAULT")
    seg_model.eval()
    input_tensor = T.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = seg_model(input_tensor)["out"]
    seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Keep only dog/cat (VOC classes 8 and 12)
    pet_mask = np.where((seg_mask == 8) | (seg_mask == 12), 1, 0).astype(int)

    # Ensure it's a list of lists of ints
    return pet_mask.tolist()

