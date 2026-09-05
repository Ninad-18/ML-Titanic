# 🚢 Titanic Survival Prediction – Machine Learning Project

This project predicts whether a passenger survived the Titanic disaster using multiple supervised machine learning models.

The goal is to understand:
- Data preprocessing
- Feature engineering
- Model comparison
- Evaluation metrics
- Model deployment using Streamlit

---

## 📊 Dataset
- **Source**: Kaggle Titanic Dataset
- **Target Variable**: `Survived` (0 = No, 1 = Yes)

---

## 🛠 Models Implemented
- Linear Regression (baseline understanding)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

---

## ⚙️ Data Preprocessing
- Missing value handling (mean / median / mode)
- Feature extraction (Deck from Cabin)
- Encoding:
  - Label Encoding (Sex)
  - One-Hot Encoding (Embarked)
- Feature Scaling (Age, Fare)
- Dropping non-numeric & irrelevant columns

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🌲 Best Performing Model
**Random Forest Classifier**
- Handles non-linearity well
- Robust to overfitting
- Strong performance on Titanic dataset

---

## 🖥 Web App (Streamlit)
A simple UI to predict survival using Random Forest.

### Run the app:
```bash
streamlit run app.py

---

Clone the repo using :- git clone https://github.com/Ninad-18/ML-Titanic
