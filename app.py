import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Semiconductor Manufacturing Test Quality Predictor")
st.write("Upload your test data CSV and select a machine learning model to evaluate chip test yields!")

# 1. SIDEBAR: Model Selection Dropdown
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Map the choice to the saved model filename
model_filenames = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "KNN": "model_knn.pkl",
    "Naive Bayes": "model_naive_bayes.pkl",
    "Support Vector Machine": "model_support_vector_machine.pkl"
}

# 2. MAIN AREA: File Uploader for Test Data
st.subheader("Upload Test Data")
uploaded_file = st.file_uploader("Upload your test data CSV (e.g., test_data.csv)", type=["csv"])

if uploaded_file is not None:
    # Load test data and preprocessing tools
    test_df = pd.read_csv(uploaded_file)
    
    if 'Target' in test_df.columns:
        X_test = test_df.drop(columns=['Target'])
        y_test = test_df['Target']
    else:
        st.error("Uploaded CSV must contain the 'Target' column for evaluation.")
        st.stop()
        
    # Load saved scaler and transform data
    scaler = joblib.load('scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    # Load selected model
    model = joblib.load(model_filenames[model_choice])
    
    # Make Predictions
    y_pred = model.predict(X_test_scaled)
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # 3. DISPLAY EVALUATION METRICS ON THE APP
    st.subheader(f"Performance Metrics for: {model_choice}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    # 4. DISPLAY CONFUSION MATRIX
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual Pass (0)", "Actual Fail (1)"], columns=["Predicted Pass (0)", "Predicted Fail (1)"])
    st.write(cm_df)
    
    st.success("Model evaluation executed successfully!")
else:
    st.info("Please upload your `test_data.csv` file using the uploader above to view metrics.")