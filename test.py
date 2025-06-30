import requests
import base64

# Replace with any image
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

res = requests.post("https://your-service.onrender.com/analyze", json={"image": img_b64})

print("Status:", res.status_code)
print("Body:", res.text)