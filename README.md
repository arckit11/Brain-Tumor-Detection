<div align="center">

# 🧠 NeuroScan AI — Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ResNet50](https://img.shields.io/badge/ResNet50-Transfer%20Learning-blueviolet?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)](#license)

> A premium **end-to-end deep learning web application** that classifies brain tumors in MRI scans into four categories (Glioma, Meningioma, Pituitary, or No Tumor) using a **fine-tuned ResNet-50** architecture. Features a state-of-the-art glassmorphism UI and dynamic confidence scoring.

</div>

---

## ⚠️ Important Disclaimers

> **Portfolio Showcase Only:** This repository is intended strictly as a portfolio display piece. The code, models, and UI designs are **not open source** and are **not open for use, distribution, or reproduction**. All rights are reserved.

> **Medical Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or medical professional for clinical decisions.

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch & Torchvision
* **Model Architecture:** ResNet-50 (Fine-tuned via Transfer Learning)
* **Backend:** Python & Flask
* **Frontend:** HTML5, CSS3 (Custom Glassmorphism Design System)
* **Data Handling:** Hugging Face Datasets API, Base64 Image Encoding

---

## 📈 Project Quantifications

* **Dataset Size:** 13,196 MRI scans across 4 distinct classes.
* **Training Time:** ~14 minutes (15 Epochs on NVIDIA RTX 4050 / 6GB VRAM).
* **Model Accuracy:** 71% Validation Accuracy (using 224x224 downsampled inputs to fit memory constraints).
* **Model Size:** 122 MB trained `.pt` weight file.
* **Optimization:** Used PyTorch Automatic Mixed Precision (AMP/FP16) reducing VRAM consumption by ~40%.

---

## ✨ Features

* **Deep Learning Powered**: Utilizes a ResNet-50 Convolutional Neural Network fine-tuned on over 13,000 MRI scans.
* **4-Class Detection**: Accurately classifies images into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.
* **Real-time Inference**: Extracts model logits and calculates a dynamic confidence percentage score.
* **Premium UI/UX**: Built with a sleek dark medical theme, frosted glass panels (glassmorphism), drag-and-drop file uploads, and smooth CSS animations.
* **Secure Image Handling**: Uploaded images are encoded in Base64 directly to memory for frontend display, meaning no junk files are saved to the server.

---

## 📊 Dataset & Model

The model was trained on a dataset of **13,196 MRI Images** sourced from Hugging Face:
- **Glioma**: 4,167 images
- **Meningioma**: 2,947 images
- **Pituitary**: 3,232 images
- **No Tumor**: 2,850 images

### Training Highlights
- **Architecture**: ResNet-50 with a custom fully connected head (`Linear -> SELU -> Dropout -> Linear -> SELU -> Dropout -> Linear -> LogSigmoid`).
- **Preprocessing**: Images are resized to `224x224`, converted to RGB, and normalized using standard ImageNet means and standard deviations.
- **Optimization**: Trained using the Adam optimizer and PyTorch Automatic Mixed Precision (AMP/FP16) for memory efficiency.

---

## 🛠️ Project Structure

```text
├── app.py                  # Main Flask application and inference logic
├── train_model.py          # PyTorch training script and dataset pipeline
├── requirements.txt        # Python dependencies
├── static/
│   └── css/
│       └── premium.css     # Custom dark-theme glassmorphism stylesheet
└── templates/
    ├── Diseasedet.html     # Landing page
    ├── uimg.html           # Drag-and-drop upload page
    └── pred.html           # Interactive results dashboard
```

---

## 🔒 License

**All Rights Reserved.**

This project is a personal portfolio piece. It is **not open source**. You may not use, copy, modify, merge, publish, distribute, sublicense, or sell copies of this software or its UI/UX designs.

<div align="center">
<i>Built with PyTorch and Flask.</i>
</div>
