# 🫁 BreatheEasy — Lung Cancer Detection CNN

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-blue?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?style=for-the-badge&logo=flask" />
</p>

> **Advanced AI Platform for Early Lung Cancer Detection Using a PyTorch ResNet18-based CNN and an Interactive Web Interface.**

---

## 🎯 Executive Summary

**BreatheEasy** leverages a custom-modified **ResNet18 Convolutional Neural Network (CNN)** to classify lung CT scan images into **6 distinct categories**. Along with the powerful PyTorch backend, the project features a **Flask REST API** and a beautifully designed, responsive **HTML/JS web application** that provides real-time predictions and educational lung cancer insights.

**Categories Classified:**
1. Normal (Healthy)
2. Benign (Non-cancerous)
3. Adenocarcinoma (NSCLC subtype)
4. Large Cell Carcinoma (NSCLC subtype)
5. Squamous Cell Carcinoma (NSCLC subtype)
6. Malignant (General malignant)

---

## 🏗️ System Architecture

### 🌐 1. Client Layer (Frontend)
- **Interface**: Responsive Web UI built with **HTML5, TailwindCSS, & pure JavaScript**.
- **Action**: User uploads a CT Scan image (`.png` or `.jpg`).
- **Transfer**: Image is sent via HTTP POST to the API Gateway.

### ⚙️ 2. API Gateway (Backend)
- **Framework**: **Flask REST API** (`app.py`).
- **Endpoint**: `/predict` securely receives the image payload.
- **Handling**: Validates the upload and saves it locally as a temporary file for processing.

### 🛠️ 3. Data Processing Pipeline
- **Tools**: **Torchvision** & **PIL** (`preprocessing.py`).
- **Transformations**:
  - Converts image to **Grayscale** (1 channel).
  - Resizes to strictly **256x256 pixels**.
  - Transforms into a normalized **PyTorch Tensor** `(1, 1, 256, 256)`.

### 🧠 4. AI Inference Engine
- **Model Architecture**: Custom-modified **ResNet18 CNN**.
- **Weights**: Loads best-performing state dictionary (`lung_model_6class_best.pth`).
- **Classification**: Performs a forward pass and applies a **Softmax** activation to calculate probabilities across the 6 cancer classes.
- **Decision**: Uses `argmax` to select the class with the highest confidence.

### 📤 5. Response Layer
- **Data Delivery**: API constructs a JSON payload containing the `prediction` (String) and `confidence score` (Percentage).
- **Cleanup**: The temporary server-side image file is permanently deleted.
- **UI Update**: The frontend dynamically renders the results to the user on a visually polished "Result Card".

---

## 🚀 Key Features

### 🧠 **AI-Powered Detection**
- **Architecture**: Transfer learning utilizing **ResNet18** with structural modifications:
  - Input layer adapted from 3 RGB channels to 1 grayscale channel for optimized CT scan processing.
  - Final fully connected layer customized to output probabilities for 6 specific lung cancer classes.
- **Robust Training Pipeline**: Includes dynamic weighted loss logic to handle dataset class imbalances and heavy data augmentation (rotation, translation, horizontal flip) to combat overfitting.

### 💻 **Interactive Web Platform**
- **Modern UI**: Tailored with **TailwindCSS**, featuring glassmorphism elements, dark/light mode toggle, and smooth custom CSS animations.
- **Client-Side Validation**: Ensures only valid image files are processed and features a carefully designed loading state.
- **Health Educational Hub**: The page directly embeds vital information about early warning signs, prevention strategies, risk factors, and actionable medical guidelines alongside a strict Medical Disclaimer.

### 🛡️ **Privacy & Security**
- **Zero Image Retention**: Uploaded scans are stored purely in a temporary state (`temp_image.png`) and are permanently purged immediately following the inference cycle.

---

## 🛠️ Tech Stack & Implementation

| **Component**      | **Technology / Library** | **Purpose**                               |
|-------------------|--------------------------|-------------------------------------------|
| **Backend API**   | Flask, Flask-CORS        | Orchestrates uploads and AI inference     |
| **Machine Learning**| PyTorch, Torchvision     | Model building, training, and deployment  |
| **Image Processing**| PIL (Pillow), OpenCV     | Medical-grade tensor transformations      |
| **Frontend UI**   | HTML5, CSS3, JavaScript  | Real-time interactivity and user feedback |
| **Analysis / Eval** | Scikit-learn, NumPy      | Confusion Matrix, Precision/Recall metrics|

---

## 📂 Project Structure

```text
├── backend/                    # Core Backend Services
│   ├── app.py                  # Main Flask API Application
│   ├── model/                  # Trained PyTorch Model Weights (.pth)
│   └── utils/                  # Utility Functions
│       ├── preprocessing.py    # Torchvision Image Preprocessing
│       └── label_utils.py      # Folder-to-Class Number Mappings
├── training/                   # Model Training Module
│   ├── model.py                # Definition of Modified ResNet18
│   ├── data_loader.py          # Custom PyTorch Dataset Loader
│   ├── train.py                # Training execution, validation loop, model saving
│   └── evaluate.py             # Script to generate classification reports
├── frontend/                   # Client-Side Interface
│   ├── index.html              # Main App Interface (Tailwind CSS)
│   └── chat.html               # AI Chatbot Medical Assistant Interface
├── dataset/                    # Directory for structured train/test/valid sets
└── requirements.txt            # Python dependencies lists
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/atharvaaa10/Lung-Cancer-Detection-using-CNN-model.git
cd Lung-Cancer-Detection-using-CNN-model

# Install the required Python dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
Start the Backend API Server:
```bash
python backend/app.py
# The Flask server will start locally at http://127.0.0.1:5000
```

Start the Frontend interface:
Simply open `frontend/index.html` in your preferred web browser, or if you prefer a local server:
```bash
cd frontend
python -m http.server 8000
# Access the web interface at http://localhost:8000
```

---

## 📊 Dataset & Training

The model was iteratively trained and validated using a structured dataset containing lung CT scans split into `train`, `valid`, and `test` sub-folders.

- **Preprocessing Parameters**:
  - Image resizing to exactly `256 x 256` pixels
  - Grayscale conversion
  - Adam Optimizer (Learning Rate: `0.001`)
  - Batch Size: `16`
  - Max Epochs: `30`
- **Imbalance Handling**: The training script automatically assigns class-specific weights to the `CrossEntropyLoss` function inverse to their statistical frequency in the dataset.

---

## ⚠️ Important Medical Disclaimer

> **This AI tool is designed strictly for educational and preliminary screening research purposes.** It cannot and should not replace professional medical consultation, diagnosis, or treatment. Diagnostic prediction models inherently possess limitations including the potential for false positive and false negative results. **Always consult a qualified healthcare provider regarding diagnostic questions.**

---
<div align="center">
<b>BreatheEasy</b> — <i>AI Technology Advancing Healthcare Awareness</i>
</div>
