import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="Semiconductor Manufacturing Quality Predictor", layout="wide")

st.title("🔬 Semiconductor Manufacturing Quality Prediction App")
st.write("This application evaluates machine learning classification performance for semiconductor test yield prediction (UCI SECOM Dataset).")

# 1. SIDEBAR: Model Selection
st.sidebar.header("⚙️ Configuration & Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

model_filenames = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "KNN": "model_knn.pkl",
    "Naive Bayes": "model_naive_bayes.pkl",
    "Support Vector Machine": "model_support_vector_machine.pkl"
}

# 2. TRAINING DATASET PERFORMANCE & SCORES PREVIEW
st.header("📊 Training Dataset Performance & Benchmark Scores")
st.write("Below is the benchmark performance summary of all 5 classification models evaluated on the training dataset during model development:")

# Model comparison table for training dataset performance
train_scores_data = {
    "ML Model Name": [
        "Logistic Regression", 
        "Decision Tree", 
        "KNN", 
        "Naive Bayes", 
        "Support Vector Machine"
    ],
    "Accuracy": [0.9322, 0.8917, 0.9338, 0.8854, 0.9338],
    "AUC": [0.6845, 0.5420, 0.6120, 0.6510, 0.6920],
    "Precision": [0.4500, 0.1818, 0.5000, 0.2250, 0.5000],
    "Recall": [0.4286, 0.1905, 0.0952, 0.4286, 0.0476],
    "F1 Score": [0.4390, 0.1860, 0.1600, 0.2951, 0.0870],
    "MCC": [0.4102, 0.1312, 0.1954, 0.2580, 0.1415]
}

train_df = pd.DataFrame(train_scores_data)

# Highlight selected model in the training score table
def highlight_selected(row):
    if row["ML Model Name"] == model_choice:
        return ['background-color: #d1e7dd; font-weight: bold'] * len(row)
    return [''] * len(row)

st.dataframe(train_df.style.apply(highlight_selected, axis=1), use_container_width=True)

st.markdown("---")

# 3. TEST DATA UPLOADER & PREDICTION EVALUATION
st.header("🧪 Test Data Evaluation & Live Prediction")
st.write("Upload your test dataset CSV file (`test_data.csv`) to calculate evaluation metrics on new, unseen data.")

uploaded_file = st.file_uploader("Upload Test Data CSV File", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    with st.expander("🔍 Preview Uploaded Test Data"):
        st.dataframe(test_df.head(10))
        
    if 'Target' in test_df.columns:
        X_test = test_df.drop(columns=['Target'])
        y_test = test_df['Target']
    else:
        st.error("Uploaded CSV must contain the 'Target' column for evaluation.")
        st.stop()
        
    try:
        # Load scaler & trained model
        scaler = joblib.load('scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        
        model = joblib.load(model_filenames[model_choice])
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0

        # Calculate metrics on test data
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Metrics display section
        st.subheader(f"📈 Evaluation Metrics on Uploaded Data for: **{model_choice}**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=["Actual Pass (0)", "Actual Fail (1)"], 
            columns=["Predicted Pass (0)", "Predicted Fail (1)"]
        )
        st.table(cm_df)
        
        st.success("Model evaluation on uploaded test data completed successfully!")
        
    except Exception as e:
        st.error(f"Error loading model files or scaler: {e}")
else:
    st.info("👆 Please upload your `test_data.csv` file above to run predictions and view metrics.")