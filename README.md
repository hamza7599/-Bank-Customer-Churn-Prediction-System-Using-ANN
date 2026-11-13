# 🏦 Bank Customer Churn Prediction System

An end-to-end Deep Learning classification project predicting bank customer churn using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras, paired with an interactive **Streamlit frontend web application** for real-time risk assessment and data exploration.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Streamlit Frontend App](#-streamlit-frontend-app)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Running in Google Colab](#-running-in-google-colab)

---

## 🔍 Overview
Customer retention is crucial in retail banking. Identifying customers at high risk of closing accounts allows retention teams to intervene proactively. This repository contains the complete machine learning pipeline—from data cleaning and preprocessing to neural network training, artifact serialization, and an intuitive user interface powered by Streamlit.

---

## ⚙️ Project Architecture & Workflow
1. **Data Cleaning:** Dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`) and checked for missing values.
2. **Stratified Train-Test Split:** Split the dataset `80/20` (`8,000` train / `2,000` test) while preserving the original churn distribution (`20.37%`).
3. **Preprocessing Pipeline (`ColumnTransformer`):**
   - **StandardScaler** applied to all numerical features.
   - **OneHotEncoder (drop='first')** applied to categorical features (`Geography`, `Gender`).
   - Fitted strictly on the training set to prevent data leakage.
4. **ANN Modeling (TensorFlow/Keras):** Feedforward neural network featuring dense layers with ReLU activations and Dropout regularization.
5. **Training & Regularization:** Trained using the Adam optimizer, binary crossentropy loss, and **EarlyStopping** (`patience=10`, `restore_best_weights=True`).
6. **Artifact Persistence:** Serialized the trained neural network model (`churn_ann_model.keras`) and preprocessing pipeline (`preprocessor.joblib`).
7. **Frontend Deployment:** Served via an interactive **Streamlit** web app (`app.py`) for live scoring.

---

## 📊 Dataset
Trained on **`Churn_Modelling.csv`** (`10,000` customer records):
- **Numerical Features:** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Categorical Features:** `Geography` (`France`, `Germany`, `Spain`), `Gender` (`Female`, `Male`)
- **Target Variable (`Exited`):** `1` = Churned, `0` = Retained

---

## 🧠 Model Performance

### Neural Network Summary
* **Input Features:** `11` (after one-hot encoding)
* **Hidden Layers:** Layer 1 (64 units, ReLU + 0.3 Dropout) → Layer 2 (32 units, ReLU + 0.2 Dropout)
* **Output Layer:** 1 unit (Sigmoid activation)
* **Total Trainable Parameters:** `2,881`

### Test Set Evaluation (`2,000` Unseen Samples)
* **Test Accuracy:** `86.05%`
* **Precision (Churn = 1):** `0.79`
* **Recall (Churn = 1):** `0.43`
* **F1-Score (Churn = 1):** `0.56`
* **Confusion Matrix:**
  * True Negatives (Stayed correctly identified): `1,546`
  * False Positives (Stayed flagged as churn): `47`
  * False Negatives (Churned missed): `232`
  * True Positives (Churned correctly identified): `175`

---

## 🖥️ Streamlit Frontend App (`app.py`)

The Streamlit web interface provides an accessible way to interact with the model:
* **Interactive Input Controls:** Sliders, dropdowns, and number fields for all customer demographic and account attributes (Age, Balance, Credit Score, Geography, etc.).
* **Real-Time Inference:** Instantly transforms inputs using the saved `preprocessor.joblib` pipeline and passes them to `churn_ann_model.keras`.
* **Visual Risk Feedback:** Displays a dynamic progress bar, numerical churn probability percentage, and clear risk alert callouts (`streamlit` success/error banners).

---

## 📁 Repository Structure
```text
├── Churn_Modelling.csv        # Raw dataset
├── churn_prediction.ipynb     # Complete training and evaluation notebook
├── churn_ann_model.keras      # Saved trained Keras ANN model
├── preprocessor.joblib        # Saved scikit-learn ColumnTransformer pipeline
├── app.py                     # Streamlit frontend web application code
├── training_history.png       # Loss and accuracy curves over epochs
├── confusion_matrix.png       # Confusion matrix evaluation heatmap
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
