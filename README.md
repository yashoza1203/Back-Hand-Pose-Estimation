# 🖐️ Back-Hand Pose Estimation

## ▶️ Demo: Best Performing Model

https://github.com/user-attachments/assets/27095bf7-7b58-4beb-9d4f-98735947d68e
⬆️ *YOLOv8n @ 224×224 – 30 FPS on CPU (ONNX Runtime)*

> 📽️ This video shows real-time back-hand keypoint estimation on CPU using the YOLOv8n model with ONNX Runtime. High precision + great speed.

---

This project explores back-hand keypoint estimation using CNN-based models trained on the **Thumb Index 1k** dataset.

---

## 📂 Models Trained

### 1. ✅ EfficientNet B0 (PyTorch)
- **Framework**: PyTorch
- **Hand Detection**: Accurate
- **Keypoint Estimation**: Poor performance
- **Inference Speed**: ~15 FPS on CPU
- **Observation**: Detects hands reliably but struggles with accurate keypoint estimation.

---

### 2. ✅ ResNet-50 (Keras)
- **Framework**: Keras
- **Hand Detection**: Accurate
- **Keypoint Estimation**: Moderate performance
- **Inference Speed**: ~1 FPS on CPU
- **Observation**: Decent accuracy, but too slow for real-time applications.

---

### 3. ✅ YOLOv8 Pose (n, s, m, x)
- **Framework**: Ultralytics YOLOv8, exported to **ONNX Runtime**
- **Image Sizes**: 224×224 and 640×640
- **Best Performance**: Models trained and exported at 224×224 resolution
- **Deployment**: ONNX Runtime for high-speed CPU inference
- **Detection**: Accurate and consistent
- **Keypoint Estimation**: Highly precise
- **Top Performer**: `YOLOv8n` @ 224×224 → **30 FPS on CPU with ONNX Runtime**
- **Observation**: Smaller image sizes improved FPS significantly without compromising keypoint accuracy.

---

## 📌 Contributing

Feel free to fork the repo, open issues, or submit PRs! Any improvements, new models, or suggestions are welcome.
