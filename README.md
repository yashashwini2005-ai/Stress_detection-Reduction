# 🧠 Stress Detection & Reduction System

## 📌 Overview
This project focuses on detecting human stress levels using physiological signals from the WESAD dataset. The system applies machine learning techniques to analyze wearable sensor data such as heart rate, electrodermal activity (EDA), temperature, and accelerometer readings.

The goal is to build an intelligent system that can automatically classify stress levels and help in stress management.

---

## 🎯 Problem Statement
Stress is a major issue in modern life, especially among students and professionals. Early detection of stress can help in:
- Preventing mental health issues
- Improving productivity
- Enabling timely intervention

This project aims to develop a machine learning model that can detect stress using physiological data.

---

## 📊 Dataset
- **Dataset Used:** WESAD (Wearable Stress and Affect Detection)
- Contains multimodal data including:
  - Heart Rate (HR)
  - Electrodermal Activity (EDA)
  - Body Temperature (TEMP)
  - Blood Volume Pulse (BVP)
  - Accelerometer Data (ACC)

> ⚠️ Dataset is not uploaded due to size.  
> Download from: https://www.kaggle.com/ (Search: WESAD Dataset)

---

## ⚙️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 🔄 Workflow
1. Data Collection (WESAD Dataset)
2. Data Preprocessing
3. Feature Extraction
4. Model Training
5. Stress Classification

---

## 🧠 Machine Learning Approach
- Data cleaning and preprocessing
- Feature extraction from physiological signals
- Model training using ML algorithms (e.g., Random Forest / Logistic Regression)
- Classification of stress levels:
  - Low Stress
  - Medium Stress
  - High Stress

Machine learning models analyze physiological patterns to classify stress levels effectively :contentReference[oaicite:0]{index=0}

---

## 📁 Project Structure
Stress_detection-Reduction/
│── S1/, S2/, S3/... (Dataset - ignored)
│── notebooks/ (if added later)
│── src/ (model code)
│── README.md
│── requirements.txt
│── .gitignore


---

## ▶️ How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yashashwini2005-ai/Stress_detection-Reduction.git
cd Stress_detection-Reduction

2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Project
python main.py

📈 Future Improvements
Real-time stress detection using sensors
Web application using FastAPI
Dashboard visualization
Deep Learning models (LSTM, CNN)
Integration with wearable devices
💡 Applications
Healthcare monitoring
Workplace stress management
Student mental health analysis
Wearable health tech systems

⭐ Conclusion

This project demonstrates how machine learning can be applied to real-world health problems like stress detection. It provides a foundation for building intelligent health monitoring systems.
