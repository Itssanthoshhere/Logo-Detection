<div align="center">

<h1>🔍 Logo Detection – Flask + Deep Learning</h1>

A modern **deep learning–powered web application** that identifies brand logos from images using a trained **Xception CNN model**.  
Built with **Flask**, **TensorFlow**, and a clean, interactive UI.

<!-- Tech Stack Badges -->

<div>
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/-HTML-E34F26?style=for-the-badge&logo=html5&logoColor=white"/>
  <img src="https://img.shields.io/badge/-CSS-1572B6?style=for-the-badge&logo=css3&logoColor=white"/>
</div>

<h3>✨ Deep Learning–Powered Brand Logo Recognition</h3>
<b>Upload an image → Predicts the brand logo with confidence.</b><br/>
Trained on the <b>Flickr Logos 27 Dataset</b>, deployed via a clean Flask interface.

</div>

---

## 📋 Table of Contents

- [📋 Table of Contents](#-table-of-contents)
- [✨ Introduction](#-introduction)
- [⚙️ Tech Stack](#️-tech-stack)
  - [🧠 Machine Learning](#-machine-learning)
  - [🌐 Backend](#-backend)
  - [🎨 Frontend](#-frontend)
- [🔋 Features](#-features)
  - [🔍 Logo Detection](#-logo-detection)
  - [🖼️ Modern UI](#️-modern-ui)
  - [🧠 Deep Learning](#-deep-learning)
  - [🗂️ Flask Integration](#️-flask-integration)
- [📁 Dataset](#-dataset)
- [🤖 Model Architecture](#-model-architecture)
- [🚀 Project Workflow](#-project-workflow)
- [🤸 Quick Start](#-quick-start)
  - [🔧 Prerequisites](#-prerequisites)
  - [1️⃣ Clone Repository](#1️⃣-clone-repository)
  - [2️⃣ Create Virtual Environment](#2️⃣-create-virtual-environment)
  - [3️⃣ Install Dependencies](#3️⃣-install-dependencies)
  - [4️⃣ Run the App](#4️⃣-run-the-app)
- [🧱 Project Structure](#-project-structure)
- [🖥️ App Flow](#️-app-flow)
- [🧠 Architecture Overview](#-architecture-overview)
  - [🧩 Backend (Flask)](#-backend-flask)
  - [🎨 Frontend](#-frontend-1)
  - [🤖 Deep Learning Model](#-deep-learning-model)
- [🚀 Future Enhancements](#-future-enhancements)
- [🤝 Contribution](#-contribution)
- [🔗 Contacts](#-contacts)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
  - [⭐ Show Your Support](#-show-your-support)

---

## ✨ Introduction

**AI Logo Detection** is an end-to-end deep learning project that identifies brand logos in images.  
This project consists of:

- A **deep learning model (Xception + custom CNN head)** trained via transfer learning.
- A **Flask web app** where users upload an image.
- The model predicts:
  - **Logo Name**
  - **Confidence Score**
  - Displays the **uploaded image**.

Built for accuracy, performance, and clean UI.

---

## ⚙️ Tech Stack

### 🧠 Machine Learning

- **TensorFlow 2.x**
- **Keras**
- **Xception (pretrained on ImageNet)**
- **OpenCV**
- **NumPy**
- **Pandas**

### 🌐 Backend

- **Flask**
- **Werkzeug**

### 🎨 Frontend

- **HTML5**
- **CSS3**
- **JavaScript (for image preview)**

---

## 🔋 Features

### 🔍 Logo Detection

- Upload any image (JPG/PNG)
- Model returns:
  - Predicted brand
  - Confidence %
  - Raw probability vector

### 🖼️ Modern UI

- Drag-and-drop upload
- Live image preview
- Clean dark theme
- Result preview panel

### 🧠 Deep Learning

- 27 logo classes
- Transfer learning using Xception
- High accuracy model stored in `logo.h5`

### 🗂️ Flask Integration

- Handles uploads
- Saves files to `static/uploads/`
- Renders predictions dynamically

---

## 📁 Dataset

This project uses the **Flickr Logos 27 Dataset**, which contains:

- 27 brand classes
- Training + validation annotations
- Bounding box coordinates
- Distractor images

Dataset includes brands such as:

```

Adidas, Apple, BMW, CocaCola, Ferrari, Ford, Google, Intel, Nike,
Pepsi, Porsche, Puma, RedBull, Starbucks, Yahoo, Vodafone, McDonalds, etc.

```

---

## 🤖 Model Architecture

Built using **Transfer Learning**:

- Backbone: **Xception (pretrained on ImageNet)**
- Custom classification head:
  - AveragePooling
  - Flatten
  - Dense (128)
  - Dropout (0.5)
  - Dense (27 softmax)

Loss: `categorical_crossentropy`
Optimizer: `Adam`

The model is exported as:

```

logo.h5

```

---

## 🚀 Project Workflow

1️⃣ **Dataset extraction** → bounding box cropping
2️⃣ **Data augmentation** using `ImageDataGenerator`
3️⃣ **Transfer learning** with Xception
4️⃣ **Model training** on 224×224 images
5️⃣ **Evaluation & classification report**
6️⃣ **Model exported** to project
7️⃣ **Flask app built** for real-time inference
8️⃣ **User uploads an image**
9️⃣ **Prediction displayed in browser**

---

## 🤸 Quick Start

### 🔧 Prerequisites

- Python 3.8+
- pip
- Virtual environment recommended

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/itssanthoshhere/Logo-Detection.git
cd Logo-Detection
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

(or manually)

```bash
pip install flask tensorflow pillow numpy
```

### 4️⃣ Run the App

```bash
python app.py
```

Navigate to:

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Upload an image → get prediction 🎉

---

## 🧱 Project Structure

```
logo_flask_full/
├── app.py
├── classes.txt
├── logo.h5
├── logo_detection_flickr27.ipynb
├── logos/
│   ├── Ferrari.jpg
│   ├── McDonald's.jpg
│   └── ford.jpg
├── static/
│   ├── style.css
│   └── uploads/
├── templates/
│   └── index.html
└── venv/
```

---

## 🖥️ App Flow

1. User opens web page
2. Uploads image (drag & drop or file picker)
3. Flask receives file
4. Model processes image
5. Prediction returned
6. UI displays:

   - Logo name
   - Confidence
   - Image preview

---

## 🧠 Architecture Overview

### 🧩 Backend (Flask)

- Loads TensorFlow model at startup
- Handles `/predict` endpoint
- Saves uploaded images
- Returns prediction + confidence

### 🎨 Frontend

- Styled HTML/CSS
- Image preview before upload
- Displays prediction results

### 🤖 Deep Learning Model

- Transfer learning
- 27-class softmax classifier
- Preprocessed with normalization

---

## 🚀 Future Enhancements

- 🔲 Bounding box logo localization
- 🔲 Support for multiple logos in one image
- 🔲 Convert model to TensorFlow Lite (TFLite)
- 🔲 Deploy on Render / Railway / AWS
- 🔲 Add API-only mode (REST endpoints)
- 🔲 Add history of predictions
- 🔲 Add brand logo icons in UI

---

## 🤝 Contribution

Contributions are welcome!

1. Fork repo
2. Create a feature branch
3. Commit changes
4. Open a PR 🎉

---

## 🔗 Contacts

- **GitHub:** [Itssanthoshhere](https://github.com/Itssanthoshhere)
- **LinkedIn:** [Santhosh VS](https://linkedin.com/in/thesanthoshvs)

---

## 📄 License

For **educational and research purposes** only.
Logos belong to their respective brands.

---

## 🙏 Acknowledgements

- Flickr Logos 27 Dataset
- TensorFlow / Keras
- Flask
- Kaggle
- OpenCV

---

### ⭐ Show Your Support

If you like this project, **give it a star** ⭐ on GitHub — it motivates me to build more awesome ML apps!

---
