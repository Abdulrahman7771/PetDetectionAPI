import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64
import cv2
import requests


# Load models once
cls_model = resnet50(weights="IMAGENET1K_V1")
cls_model.eval()

yolo_model = YOLO("yolov5su.pt")

seg_model = deeplabv3_resnet50(weights="DEFAULT")
seg_model.eval()

imagenet_labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

segmentation_labels = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def read_image_from_base64(base64_string):
    image_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def classify(image: Image.Image) -> bool:
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cls_model(input_tensor)
        pred = output.argmax(dim=1).item()
        label = imagenet_labels[pred].lower()
        print("Classification Label:", label)
        return "dog" in label or "cat" in label

def detect(image: Image.Image):
    results = yolo_model.predict(image, conf=0.5)[0]
    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        label = yolo_model.model.names[cls]
        if label.lower() in ["cat", "dog"]:
            coords = box.xyxy[0].tolist()
            boxes.append(coords)
    return boxes

def segment(image: Image.Image):
    input_tensor = T.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = seg_model(input_tensor)["out"]
    seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    pet_mask = (seg_mask == 8).astype(np.uint8)  # 8 is 'cat'
    pet_mask += (seg_mask == 12).astype(np.uint8)  # 12 is 'dog'
    return pet_mask.tolist()  # convert to regular list so JSON can return it

