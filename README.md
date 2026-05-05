# 📉 Customer Churn Prediction using Deep Learning

## 📌 Project Overview
This project focuses on predicting customer churn using deep learning techniques. The goal is to identify customers who are likely to leave a service, enabling businesses to take proactive retention actions.

An **Artificial Neural Network (ANN)** model is built to analyze customer behavior and predict churn with high accuracy.

---

## 🎯 Objectives
- Predict whether a customer will churn or not  
- Identify key factors influencing churn  
- Help businesses improve customer retention strategies  

---

## 📊 Dataset
- Source: Kaggle (Telco Customer Churn Dataset)  
- Features include:
  - Customer demographics  
  - Account information (tenure, contract type)  
  - Services subscribed  
  - Monthly and total charges  

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy** – Data preprocessing  
- **Matplotlib, Seaborn** – Data visualization  
- **TensorFlow / Keras** – Deep Learning model  
- **Scikit-learn** – Data splitting & evaluation  

---

## ⚙️ Project Workflow
1. Data Collection & Loading  
2. Data Cleaning & Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering & Encoding  
5. Model Building using ANN  
6. Model Training & Validation  
7. Model Evaluation  
8. Prediction & Insights  

---

## 🤖 Model Details
- Model: **Artificial Neural Network (ANN)**
- Layers:
  - Input Layer  
  - Hidden Layers (Dense + Activation)  
  - Output Layer (Sigmoid for binary classification)  

- Why Deep Learning?
  - Captures complex patterns in customer behavior  
  - Improves prediction accuracy over traditional models  

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 📊 Key Insights
- Customers with **short tenure** are more likely to churn  
- **Month-to-month contracts** have higher churn rates  
- Higher **monthly charges** increase churn probability  

---

## 🚀 Results
The deep learning model successfully identified churn-prone customers and achieved strong classification performance, making it useful for real-world retention strategies.

---

## 💡 Future Improvements
- Hyperparameter tuning for better performance  
- Use advanced architectures (LSTM for sequential behavior)  
- Deploy model using Streamlit or Flask  
- Integrate real-time prediction system  

---

## 📁 Project Structure
├── data/
├── notebooks/
├── model/
├── app/
├── README.md


---

## 🧑‍💻 Author
Aman Singh  
B.Tech CSE (Data Science)  

---

## ⭐ Acknowledgements
- Kaggle for dataset  
- TensorFlow & Keras community  

---

## 📌 How to Run
```bash
git clone https://github.com/your-username/churn-prediction-deep-learning.git
cd churn-prediction-deep-learning
pip install -r requirements.txt
python main.py
