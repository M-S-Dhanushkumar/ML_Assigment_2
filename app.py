import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Page Title
# ----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Upload a test dataset OR manually enter patient details.")

# ----------------------------
# Load Model & Scaler
# ----------------------------
scaler = joblib.load("models/scaler.pkl")

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model = joblib.load(f"models/{model_choice}.pkl")

# ----------------------------
# IMPORTANT — Training Column Order
# (Must match your model training)
# ----------------------------
training_columns = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "serum_cholestoral",
    "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "ST_segment",
    "major_vessels",
    "thal"
]

# ----------------------------
# Column Mapping (Your dataset → Training names)
# ----------------------------
column_mapping = {
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "serum_cholestoral",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_electrocardiographic_results",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "slope": "ST_segment",
    "ca": "major_vessels"
}

# ----------------------------
# Function to Fix Columns
# ----------------------------
def preprocess_data(df):
    df = df.rename(columns=column_mapping)
    df = df[training_columns]   # ensure correct order
    return df

# OPTION 1 — Upload CSV
# ----------------------------
st.header("Option 1: Upload Test CSV")

uploaded_file = st.file_uploader("heart.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")
    st.dataframe(df.head())

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Fix column names
    X = preprocess_data(X)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)