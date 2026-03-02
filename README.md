# 🦴 OsteoAI: Knee Osteoarthritis Severity Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url-here.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

## 📊 Executive Summary
This project bridges the gap between deep learning and healthcare administration. OsteoAI is an interactive web application designed to assist clinical workflows by automatically predicting the severity of Knee Osteoarthritis from radiographic images (X-rays). 

Developed as part of the **Deep Learning for Managers** curriculum, this tool demonstrates how Convolutional Neural Networks (CNNs) can be deployed to optimize medical triage, allocate resources efficiently, and provide data-driven second opinions in high-volume clinical settings.

## 🎯 Business & Managerial Value
* **Optimized Triage:** Automates the initial screening process, allowing radiologists to prioritize severe (Grade 3 & 4) cases.
* **Cost Efficiency:** Reduces the time spent on manual diagnosis for healthy or mildly arthritic scans.
* **Data-Driven Insights:** Provides quantitative confidence scores alongside categorical predictions to support evidence-based clinical decision-making.

## 🧠 Technical Architecture
The core engine is built using **Transfer Learning**. We utilized the pre-trained **MobileNetV2** architecture, freezing its base layers and appending custom dense layers optimized for a 5-class classification problem. 

* **Dataset:** [Knee Osteoarthritis Dataset with Severity Scoring](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Kellgren-Lawrence grading scale).
* **Framework:** TensorFlow / Keras
* **Deployment & UI/UX:** Streamlit Community Cloud
* **Data Visualization:** Plotly (Interactive Gauge Charts) & Lottie (Fluid Animations)

## ⚙️ Local Installation & Usage

To run this dashboard locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/osteoai-managerial-dashboard.git](https://github.com/your-username/osteoai-managerial-dashboard.git)
   cd osteoai-managerial-dashboard
