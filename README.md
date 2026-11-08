# ML_Assignment2
# Semiconductor Manufacturing Test Quality Prediction

## 1. Project Overview & Problem Description
In modern semiconductor fabrication, manufacturing processes involve hundreds of continuous steps with thousands of sensors monitoring operational parameters in real time. Yield loss—where chips fail end-of-line functional tests—is extremely costly. 

The goal of this project is to build an automated machine learning workflow capable of predicting semiconductor chip test failure (`Pass` vs. `Fail`) based on high-dimensional sensor telemetry data. Using the public **UCI SECOM** dataset, we clean missing values, perform statistical feature selection to remove noisy/dead sensors, train five distinct machine learning models, and deploy an interactive **Streamlit Web Application** for real-time model evaluation and yield prediction.

---

## 2. Dataset Description
* **Dataset Name:** UCI SECOM (Semiconductor Manufacturing Data)
* **Total Records:** 1,567 instances
* **Total Raw Features:** 590 numerical sensor measurement signals + 1 timestamp
* **Target Label (`Pass/Fail`):**
  * `0` (Mapped from `-1`): **Pass** (Normal operation, ~1,463 samples)
  * `1` (Mapped from `1`): **Fail** (Defective yield loss, ~104 samples)
* **Dataset Characteristics:** Highly imbalanced class distribution, presence of missing values (nulls), and high dimensionality (over 500 features).

---

## 3. Data Preprocessing & Feature Selection
To optimize model performance, reduce noise, and solve the high-dimensionality bottleneck ("Curiosity of Dimensionality"), the following pipeline was executed:
1. **Timestamp Removal:** Dropped non-predictive `Time` tags.
2. **Variance Filtering (`VarianceThreshold`):** Identified and dropped zero-variance columns (sensors that remained static across all samples).
3. **Missing Value Imputation (`SimpleImputer`):** Calculated column-wise arithmetic means to handle missing sensor values.
4. **Feature Selection (`SelectKBest`):** Applied ANOVA F-value analysis (`f_classif`) to extract the **top 30 most informative features** strongly correlated with yield failures.
5. **Standardization (`StandardScaler`):** Applied mean-centering and standard scaling so distance-based algorithms (SVM, KNN) function effectively.

---

## 4. Machine Learning Model Performance Comparison

Five supervised classification models were trained on 80% of the data and evaluated on the remaining 20% unseen test split. 

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9204 | 0.6908 | 0.2222 | 0.1429 | 0.1739 | 0.1340 |
| **Decision Tree Classifier** | 0.8662 | 0.5097 | 0.0526 | 0.0714 | 0.0606 | -0.0205 |
| **K-Nearest Neighbors (KNN)** | 0.9522 | 0.5960 | 0.0000 | 0.0000 | 0.0000 | -0.0233 |
| **Naive Bayes (GaussianNB)** | 0.8917 | 0.6401 | 0.1875 | 0.2143 | 0.2000 | 0.1419 |
| **Support Vector Machine (SVM)** | 0.9554 | 0.6588 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Key Observations & Metrics Analysis
1. **High Baseline Accuracy vs. Imbalance:** Standard Accuracy appears artificially high (~95%) for models like SVM and KNN because over 93% of the samples belong to the `Pass` class. Evaluating models based on Accuracy alone is misleading for imbalanced industrial data.
2. **Best Overall Classifier:** **Logistic Regression** and **Naive Bayes** achieved the best balance of **AUC (0.6908 and 0.6401)**, **Recall**, and **Matthews Correlation Coefficient (MCC)**, proving effective at picking up subtle signals for rare failure cases without overfitting to the majority class.
3. **MCC & F1 Importance:** MCC provides the most realistic indicator of model effectiveness on this dataset, penalizing models that simply predict all chips as `Pass`.

---

## 5. Repository Structure
```text
├── app.py                      # Main Streamlit web application
├── train_and_evaluate.py       # Model training, feature selection & evaluation script
├── uci-secom.csv              # Raw input dataset
├── test_data.csv               # Preprocessed test split CSV for Streamlit testing
├── imputer.pkl                 # Fitted mean imputer object
├── selector.pkl                # Fitted SelectKBest feature selector
├── scaler.pkl                  # Fitted StandardScaler object
├── model_logistic_regression.pkl
├── model_decision_tree.pkl
├── model_knn.pkl
├── model_naive_bayes.pkl
├── model_support_vector_machine.pkl
├── requirements.txt            # Python dependencies for deployment
└── README.md                   # Assignment documentation and analysis
```

---

## 6. Local Setup and Execution Instructions

### Prerequisites
* Python 3.9+
* `pip` package manager

### Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models & Generate Saved Artifacts (Optional):**
   ```bash
   python train_and_evaluate.py
   ```

4. **Launch the Streamlit Web App locally:**
   ```bash
   streamlit run app.py
   ```
5. Open browser at `http://localhost:8501`, upload `test_data.csv`, and select a model from the sidebar to inspect real-time performance predictions and confusion matrices.

---

## 7. Live Deployment
* **Streamlit Community Cloud Link:** `https://<your-app-subdomain>.streamlit.app`
* **GitHub Repository:** `https://github.com/<your-username>/<your-repo-name>`
