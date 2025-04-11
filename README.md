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
- **Detection**: Accurate and consistent
- **Keypoint Estimation**: Highly precise
- **Top Performer**: `YOLOv8n` @ 224×224 → **30 FPS on CPU**
- **Observation**: Smaller image sizes improved FPS significantly without compromising keypoint accuracy.

---

## 🔍 Next Steps / To-Do

To make this project even more robust, consider adding:

- 📊 **Evaluation Metrics**: PCK, mAP, or custom accuracy metrics for keypoint localization.
- 📁 **Dataset Overview**: Size, annotation format, pose diversity, and augmentation details.
- 🏋️ **Training Info**: Epochs, batch size, learning rate, optimizer, training time, and hardware specs.
- 🖼️ **Visual Outputs**: Example images showing predicted keypoints vs. ground truth.
- 📉 **Error Analysis**: Where models fail—occlusions, low light, extreme poses, etc.
- 📋 **Model Comparison Table**: Parameters, FPS, accuracy, model size, etc.
- 🛠️ **Applications**: Real-time gesture tracking, human-computer interaction, etc.

---

## 📌 Contributing

Feel free to fork the repo, open issues, or submit PRs! Any improvements, new models, or suggestions are welcome.

---

## 📄 License

This project is licensed under [MIT License](LICENSE).
