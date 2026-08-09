<div align="center">

# 🧠 NeuroScan AI — Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ResNet50](https://img.shields.io/badge/ResNet50-Transfer%20Learning-blueviolet?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](LICENSE)

> A premium **end-to-end deep learning web application** that classifies brain tumors in MRI scans into four categories (Glioma, Meningioma, Pituitary, or No Tumor) using a **fine-tuned ResNet-50** architecture. Features a state-of-the-art glassmorphism UI and dynamic confidence scoring.

</div>

---

## ⚠️ Medical Disclaimer

> **This tool is for educational and research purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or medical professional for clinical decisions.

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

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NeuroScan-AI.git
cd NeuroScan-AI
```

### 2. Install Dependencies
Ensure you have Python 3.7+ installed. Install the required packages:
```bash
pip install -r requirements.txt
pip install torch torchvision
```

### 3. Generate the Model (Important!)
Due to GitHub's 100MB file size limits, the 122MB trained model (`bt_resnet50_model.pt`) is **not included** in this repository. 

To run the web app, you must first train the model yourself (or download the dataset and run the training script). 
```bash
# This will download the dataset, train the model, and save it to models/bt_resnet50_model.pt
python train_model.py
```
*(Training takes approximately 15-20 minutes on an NVIDIA RTX 4050 GPU).*

### 4. Run the Web App
Once the model is generated and saved in the `models/` directory, start the Flask server:
```bash
python app.py
```
The app will be live at `http://127.0.0.1:5000/`.

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
<div align="center">
<i>Built with PyTorch and Flask.</i>
</div>
