# 🖐️ Back-Hand Pose Estimation

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
- **Framework**: Ultralytics YOLOv8
- **Image Sizes**: 224×224 and 640×640
- **Best Performance**: Models trained with 224×224 images
- **Detection**
