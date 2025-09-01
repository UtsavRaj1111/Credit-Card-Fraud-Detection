# Credit Card Fraud Detection  

This project aims to detect fraudulent credit card transactions using **Machine Learning**. It applies **anomaly detection techniques** (Isolation Forest & Local Outlier Factor) and trains an **XGBoost classifier** on a balanced dataset created with **SMOTE**. The system is deployed as a **Streamlit web application** for interactive predictions.  

---

## 🚀 Features  
- Preprocessing with **StandardScaler**  
- Dataset balancing using **SMOTE**  
- Fraud detection insights using **Isolation Forest** & **Local Outlier Factor**  
- XGBoost classifier for prediction  
- ROC Curve visualization for evaluation  
- **Streamlit Web App** for user interaction  
- Input fields for `V1`, `V2`, and `Amount`  
- Clean **UI with light-dark theme customization**  

---

## 📂 Project Structure  
ChatGPT said:

Got it ✅ Here’s the exact README.md file you can place in your project folder:

# Credit Card Fraud Detection  

This project aims to detect fraudulent credit card transactions using **Machine Learning**. It applies **anomaly detection techniques** (Isolation Forest & Local Outlier Factor) and trains an **XGBoost classifier** on a balanced dataset created with **SMOTE**. The system is deployed as a **Streamlit web application** for interactive predictions.  

---

## 🚀 Features  
- Preprocessing with **StandardScaler**  
- Dataset balancing using **SMOTE**  
- Fraud detection insights using **Isolation Forest** & **Local Outlier Factor**  
- XGBoost classifier for prediction  
- ROC Curve visualization for evaluation  
- **Streamlit Web App** for user interaction  
- Input fields for `V1`, `V2`, and `Amount`  
- Clean **UI with light-dark theme customization**  

---

## 📂 Project Structure  


Credit Card Fraud Detection/
│── train_model.py # Script to train the ML model
│── app.py # Streamlit web application
│── creditcard.csv # Dataset (from Kaggle)
│── model.pkl # Trained ML model
│── scaler.pkl # Scaler for preprocessing

## ⚙️ Installation  

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

Install dependencies:

pip install -r requirements.txt

## Requirements.txt

> pandas
> numpy
> scikit-learn
> imbalanced-learn
> xgboost
> matplotlib
> streamlit
> joblib

📊 Training the Model

python train_model.py

🖥️ Running the Web App

streamlit run app.py

📈 Example Output

Confusion Matrix: Shows correct vs incorrect predictions

ROC Curve: Displays model performance

Web App: Clean UI for real-time predictions

📦 Dataset

Dataset: Kaggle - Credit Card Fraud Detection

Contains:

Time, V1 ... V28, Amount, and Class (target)

Highly imbalanced: only 0.172% fraud cases

🎯 Future Improvements

Add more features in the UI (all V1-V28)

Try deep learning models (Autoencoders, LSTMs)

Deploy on cloud (Heroku, AWS, etc.)



