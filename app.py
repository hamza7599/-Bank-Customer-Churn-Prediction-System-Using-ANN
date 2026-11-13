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
    default_path = '/content/Churn_Modelling.csv'
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

# Setup page title and navigation tabs
st.title("🏦 Bank Customer Churn Center")
st.markdown("Manage customer retention with interactive analytics and predictive artificial neural networks.")

tab1, tab2 = st.tabs(["📊 Interactive Analytics Dashboard", "🔮 Predict Customer Churn"])

with tab1:
    st.header("Data Exploration & Visualizations")
    
    # Dataset source selection
    data_source = st.radio("Choose Data Source:", ["Use Default Bank Dataset", "Upload custom CSV file"], horizontal=True)
    df = None
    
    if data_source == "Upload custom CSV file":
        uploaded_file = st.file_uploader("Upload Churn_Modelling.csv", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        df = load_default_data()
        if df is None:
            st.warning("Default file '/content/Churn_Modelling.csv' was not found. Please upload a dataset to proceed.")

    if df is not None:
        # Clean identifiers if they exist in dataset
        cols_to_drop = [c for c in ['RowNumber', 'CustomerId', 'Surname'] if c in df.columns]
        df_display = df.drop(columns=cols_to_drop) if cols_to_drop else df
        
        # Summary indicators
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Customers", f"{len(df):,}")
        if 'Exited' in df.columns:
            churn_pct = (df['Exited'].mean() * 100)
            col_m2.metric("Average Churn Rate", f"{churn_pct:.2f}%")
            col_m3.metric("Retained Rate", f"{100 - churn_pct:.2f}%")
        else:
            col_m2.metric("Average Churn Rate", "N/A")
            col_m3.metric("Retained Rate", "N/A")
            
        with st.expander("🔍 Inspect Raw Dataset Summary"):
            st.dataframe(df_display.head(10))
            st.write("Dataset statistics:")
            st.dataframe(df_display.describe())
            
        # Plotting options
        st.subheader("🛠️ Custom Plot Studio")
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatter Plot", "Churn Rate Analysis"])
        
        num_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        if plot_type == "Histogram":
            selected_col = st.selectbox("Variable to Plot", num_cols)
            bins = st.slider("Number of bins", 5, 100, 25)
            sns.histplot(data=df_display, x=selected_col, hue='Exited' if 'Exited' in df_display.columns else None, 
                         kde=True, bins=bins, multiple="stack", palette="viridis", ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
            
        elif plot_type == "Boxplot":
            selected_col = st.selectbox("Numerical Variable (Y-axis)", num_cols)
            x_col = st.selectbox("Grouping Variable (X-axis)", ['Exited', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'] if 'Exited' in df_display.columns else cat_cols)
            sns.boxplot(data=df_display, x=x_col, y=selected_col, palette="Set2", ax=ax)
            ax.set_title(f"{selected_col} grouped by {x_col}")
            st.pyplot(fig)
            
        elif plot_type == "Scatter Plot":
            col_x = st.selectbox("X-axis Variable", num_cols, index=0)
            col_y = st.selectbox("Y-axis Variable", num_cols, index=min(1, len(num_cols)-1))
            sns.scatterplot(data=df_display, x=col_x, y=col_y, hue='Exited' if 'Exited' in df_display.columns else None, 
                            palette="coolwarm", alpha=0.7, ax=ax)
            ax.set_title(f"{col_y} vs {col_x}")
            st.pyplot(fig)
            
        elif plot_type == "Churn Rate Analysis":
            if 'Exited' in df_display.columns:
                group_col = st.selectbox("Analyse Churn Rate by:", ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])
                churn_by_group = df_display.groupby(group_col)['Exited'].mean().reset_index()
                churn_by_group['Exited'] = churn_by_group['Exited'] * 100
                sns.barplot(data=churn_by_group, x=group_col, y='Exited', palette="rocket", ax=ax)
                ax.set_ylabel("Churn Rate (%)")
                ax.set_title(f"Churn Rate Percentage by {group_col}")
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1), 
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                st.pyplot(fig)
            else:
                st.error("The dataset does not contain an 'Exited' column to calculate churn rate.")

with tab2:
    st.header("Single Customer Prediction Form")
    if model is None or preprocessor is None:
        st.warning("Model assets are missing. Prediction tab is disabled.")
    else:
        st.markdown("Enter customer parameters to determine probability of churn.")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            geography = st.selectbox("Geography Location", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 38)
            credit_score = st.slider("Credit Score", 350, 850, 600)
            tenure = st.slider("Tenure (Years)", 0, 10, 5)
            
        with col_right:
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
            num_products = st.selectbox("Number of Bank Products", [1, 2, 3, 4], index=0)
            has_credit_card = st.radio("Has a Credit Card?", ["Yes", "No"], index=0)
            is_active_member = st.radio("Is Active Member?", ["Yes", "No"], index=0)
            salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=85000.0, step=500.0)
            
        # Prepare input data
        input_dict = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_credit_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': salary
        }
        
        input_df = pd.DataFrame([input_dict])
        
        if st.button("Run Predictive ANN Model"):
            processed_input = preprocessor.transform(input_df)
            pred_prob = model.predict(processed_input)[0][0]
            
            st.subheader("Prediction Output Result:")
            col_res1, col_res2 = st.columns(2)
            
            col_res1.metric("Probability of Churn", f"{pred_prob * 100:.2f}%")
            
            if pred_prob >= 0.5:
                col_res2.error("🔴 RISK: This customer is highly likely to churn.")
            else:
                col_res2.success("🟢 SAFE: This customer is expected to stay with the bank.")
