import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure page style and layout
st.set_page_config(page_title="Bank Churn Portal & Dashboard", layout="wide", initial_sidebar_state="expanded")

# Helper function to load model and preprocessor safely
@st.cache_resource
def load_model_resources():
    try:
        model = tf.keras.models.load_model('churn_ann_model.keras')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None

model, preprocessor = load_model_resources()

# Load default dataset safely for the visualization dashboard
@st.cache_data
def load_default_data():
    default_path = 'Churn_Modelling.csv'
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

df_data = load_default_data()

# App Title & Layout
st.title("🏦 Bank Customer Churn Prediction & Analytics Portal")
st.markdown("Use the navigation tabs below to explore customer analytics or predict individual churn risk.")

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Interactive Analytics Dashboard", "🔮 Predict Customer Churn"])

with tab1:
    st.subheader("Customer Exploratory Data Analysis")
    if df_data is not None:
        st.write(f"Dataset loaded successfully with **{df_data.shape[0]}** rows and **{df_data.shape[1]}** columns.")
        
        # Display dataset preview
        if st.checkbox("Show Raw Data Preview"):
            st.dataframe(df_data.head(100))
        
        col_a, col_b = st.columns(2)
        with col_a:
            x_col = st.selectbox("Select X-axis Feature", df_data.columns, index=df_data.columns.get_loc("Age"))
        with col_b:
            hue_col = st.selectbox("Group By (Hue)", ["Exited", "Geography", "Gender", "HasCrCard", "IsActiveMember"], index=0)
            
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df_data, x=x_col, hue=hue_col, multiple="dodge", kde=True, ax=ax, palette="viridis")
        plt.title(f"Distribution of {x_col} by {hue_col}")
        st.pyplot(fig)
    else:
        st.warning("Churn_Modelling.csv not found in the root directory.")

with tab2:
    st.subheader("Single Customer Prediction Form")
    
    if model is None or preprocessor is None:
        st.error("⚠️ Model assets are missing. Prediction tab is disabled until `churn_ann_model.keras` and `preprocessor.joblib` are properly loaded.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", min_value=18, max_value=92, value=40)
            tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

        with col2:
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=60000.0, step=500.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=75000.0, step=1000.0)

        if st.button("Predict Churn Risk", type="primary"):
            input_df = pd.DataFrame([{
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': 1 if has_cr_card == "Yes" else 0,
                'IsActiveMember': 1 if is_active == "Yes" else 0,
                'EstimatedSalary': estimated_salary
            }])

            processed_input = preprocessor.transform(input_df)
            churn_prob = float(model.predict(processed_input, verbose=0)[0][0])

            st.markdown("---")
            st.subheader("Prediction Result")
            st.progress(churn_prob)
            st.metric(label="Churn Probability", value=f"{churn_prob * 100:.2f}%")

            if churn_prob >= 0.5:
                st.error("⚠️ High Risk: This customer is likely to LEAVE the bank.")
            else:
                st.success("✅ Low Risk: This customer is likely to STAY with the bank.")
