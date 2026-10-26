# ❤️ Heart Disease Prediction App  

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)  
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20RandomForest%20%7C%20LogReg-blue)  
![Python](https://img.shields.io/badge/Made%20With-Python-green?logo=python)  

## 📌 Overview  
This project is a **Machine Learning powered web application** that predicts whether a person is at risk of **heart disease** based on clinical features like age, cholesterol, chest pain type, and blood pressure.  

It uses three ML models for comparison:  
- Logistic Regression  
- Random Forest  
- XGBoost ✅ *(Best Performing)*  

The app is deployed using **Streamlit**, making it interactive and easy to use.  

---

## 🚀 Features  
- 📊 **Exploratory Data Analysis (EDA)** with visualizations  
- 🤖 **Multiple ML Models** (LogReg, RF, XGBoost)  
- 🔍 **Prediction with Probability Score**  
- 🧾 **Feature Importance Analysis**  
- 🌐 **Deployed with Streamlit Cloud**  

---

## 🗂️ Project Structure  
```
heart-disease-prediction/
│── app.py                         # Streamlit App
│── complete_heart_disease_notebook.ipynb   # Training & EDA
│── heart.csv                      # Dataset
│── best_model.pkl                 # Trained Model (XGBoost)
│── scaler.pkl                     # StandardScaler for inputs
│── requirements.txt               # Dependencies
│── README.md                      # Project Documentation
```

---

## ⚙️ Installation (Run Locally)  

1. Clone the repository:  
   ```bash
   git clone https://github.com/USERNAME/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create a virtual environment (optional but recommended):  
   ```bash
   python -m venv venv
   venv\Scripts\activate   # for Windows
   source venv/bin/activate   # for Mac/Linux
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:  
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment  
The app is deployed using **[Streamlit Community Cloud](https://share.streamlit.io/)**.  

Once deployed, you’ll get a public link like:  
```
https://USERNAME-heart-disease-prediction.streamlit.app
```

---

## 🖼️ Screenshots  

### 🏠 Home Page  
*(User-friendly interface to input patient details)*  

### 📊 Prediction Result  
*(Shows whether the patient is at High Risk / Low Risk with probability %)*  

---

## 📚 Dataset  
The dataset used is the **UCI Heart Disease dataset**, which contains 303 records with 14 clinical attributes.  

---

## 👨‍💻 Author  
Developed by Himavarsha
📧 Email: varshanootheti@email.com  
🌐 Portfolio / LinkedIn / GitHub (optional)  

---

⚡ *This project is for educational purposes and not a substitute for professional medical advice.*  
