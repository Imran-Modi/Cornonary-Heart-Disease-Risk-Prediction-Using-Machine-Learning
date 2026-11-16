import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Main Title
st.title("❤️ Heart Disease Risk Prediction System")

st.caption(
    "Machine Learning Based 10-Year Coronary Heart Disease Prediction"
)

# Feature Configuration

FEATURES = [
    "Gender",
    "Age",
    "Education_Level",
    "Current_Smoker",
    "Cigarettes_Per_Day",
    "BP_Medication",
    "Previous_Stroke",
    "Hypertension",
    "Diabetes",
    "Total_Cholesterol",
    "Systolic_BP",
    "Diastolic_BP",
    "Body_Mass_Index",
    "Heart_Rate",
    "Blood_Sugar_Level"
]

NUMERICAL_COLS = [
    "Age",
    "Cigarettes_Per_Day",
    "Total_Cholesterol",
    "Systolic_BP",
    "Diastolic_BP",
    "Body_Mass_Index",
    "Heart_Rate",
    "Blood_Sugar_Level"
]

CATEGORICAL_COLS = [
    "Gender",
    "Education_Level",
    "Current_Smoker",
    "BP_Medication",
    "Previous_Stroke",
    "Hypertension",
    "Diabetes"
]
# Load Saved Machine Learning Models

@st.cache_resource
def load_artifacts():
    return joblib.load("coronary_heart_disease_models.pkl")


try:
    artifacts = load_artifacts()

    models = artifacts["models"]
    scores = artifacts["comparison"]

    median_imputer = artifacts["numerical_imputer"]
    mode_imputer = artifacts["categorical_imputer"]
    median_cigs = artifacts["median_cigs"]

    df = artifacts["dataset"]

    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    st.success("All trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
# Prepare Model Predictions

predictions = {}
probabilities = {}

for name, model in models.items():

    predictions[name] = model.predict(X_test)

    probabilities[name] = model.predict_proba(X_test)[:, 1]


# Sidebar

with st.sidebar:

    st.header("Project Settings")

    st.write("Heart Disease Classification")

    st.markdown("---")

    st.write("Available Machine Learning Models")

    for model_name in models.keys():
        st.write("•", model_name)

    st.markdown("---")

    st.info(
        "This application is for educational purposes only."
    )


# Create Navigation Tabs

tab_overview, tab_compare, tab_predict, tab_data, tab_download = st.tabs(
    [
        "Overview",
        "Compare Models",
        "Predict",
        "Explore Data",
        "Save Model"
    ]
)
# TAB 1: OVERVIEW

with tab_overview:

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")

    c2.metric("Total Features", len(FEATURES))

    c3.metric("ML Models", len(models))

    c4.metric("Target Variable", "10-Year CHD")

    st.markdown("---")

    st.write("**Dataset:** Framingham Heart Study")

    st.write(
        "**Target:** Ten_Year_CHD_Risk "
        "(0 = No recorded CHD event, 1 = CHD event recorded)"
    )

    st.subheader("Target Class Distribution")

    class_counts = df["Ten_Year_CHD_Risk"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["0 - No CHD event", "1 - CHD event"],
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ],
        color=["#3498db", "#e74c3c"]
    )

    ax.set_ylabel("Number of Records")

    ax.set_title("Heart Disease Class Distribution")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Machine Learning Algorithms")

    for model_name in models.keys():
        st.write("•", model_name)

    st.warning(
        "This application is for educational purposes only. "
        "It is not a clinically validated diagnostic tool."
    )
# TAB 2: COMPARE MODELS

with tab_compare:

    st.subheader("Model Performance Comparison")

    # Display performance metrics
    st.dataframe(
        scores.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Select evaluation metric
    metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    )

    sorted_scores = scores.sort_values(
        metric,
        ascending=False
    )

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=sorted_scores,
        x=metric,
        y="Model",
        ax=ax,
        palette="viridis"
    )

    ax.set_xlim(0, 1)
    ax.set_title(f"Model Comparison Based on {metric}")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Individual model inspection
    st.subheader("Inspect Individual Model")

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        key="inspect_model"
    )

    left, right = st.columns(2)

    # Confusion Matrix
    with left:

        cm = confusion_matrix(
            y_test,
            predictions[selected_model],
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
            f"{selected_model} - Confusion Matrix"
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

    # ROC Curve
    with right:

        fig, ax = plt.subplots(figsize=(5, 4))

        for name, probability in probabilities.items():

            RocCurveDisplay.from_predictions(
                y_test,
                probability,
                name=name,
                ax=ax
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray"
        )

        ax.set_title("ROC Curve Comparison")

        st.pyplot(fig)
        plt.close(fig)

    # Classification Report
    st.subheader("Classification Report")

    st.code(
        classification_report(
            y_test,
            predictions[selected_model],
            zero_division=0
        ),
        language="text"
    )
# TAB 3: PREDICT

with tab_predict:

    st.subheader("Heart Disease Prediction")

    st.write(
        "Enter the patient information below "
        "to generate a model classification."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Personal Information")

            gender = st.selectbox(
                "Gender",
                [0, 1],
                format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=45
            )

            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4]
            )

            smoker = st.selectbox(
                "Current Smoker",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            cigarettes = st.number_input(
                "Cigarettes Per Day",
                min_value=0.0,
                max_value=100.0,
                value=0.0
            )

            bp_meds = st.selectbox(
                "BP Medication",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            stroke = st.selectbox(
                "Previous Stroke",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with col2:

            st.markdown("### Health Measurements")

            diabetes = st.selectbox(
                "Diabetes",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            cholesterol = st.number_input(
                "Total Cholesterol",
                min_value=50.0,
                max_value=700.0,
                value=200.0
            )

            systolic = st.number_input(
                "Systolic Blood Pressure",
                min_value=50.0,
                max_value=300.0,
                value=120.0
            )

            diastolic = st.number_input(
                "Diastolic Blood Pressure",
                min_value=30.0,
                max_value=200.0,
                value=80.0
            )

            bmi = st.number_input(
                "Body Mass Index",
                min_value=10.0,
                max_value=80.0,
                value=25.0
            )

            heart_rate = st.number_input(
                "Heart Rate",
                min_value=30.0,
                max_value=250.0,
                value=75.0
            )

            glucose = st.number_input(
                "Blood Sugar Level",
                min_value=30.0,
                max_value=500.0,
                value=100.0
            )

        selected_model = st.selectbox(
            "Choose Prediction Model",
            list(models.keys()),
            key="prediction_model"
        )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True
        )

    if submitted:

        if smoker == 0:
            cigarettes = 0.0

        if smoker == 1 and cigarettes == 0:
            st.error(
                "Enter cigarettes per day for a current smoker."
            )
            st.stop()

        input_data = pd.DataFrame(
            [[
                gender,
                age,
                education,
                smoker,
                cigarettes,
                bp_meds,
                stroke,
                hypertension,
                diabetes,
                cholesterol,
                systolic,
                diastolic,
                bmi,
                heart_rate,
                glucose
            ]],
            columns=FEATURES
        )

        model = models[selected_model]

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0, 1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "Model output: Class 1 - CHD event classification"
            )

        else:

            st.success(
                "Model output: Class 0 - No CHD event classification"
            )

        st.write(
            f"Estimated model probability for class 1: "
            f"**{probability:.2%}**"
        )

        st.progress(float(probability))

        st.caption(
            "This is an educational model prediction, "
            "not a clinically validated personal risk estimate."
        )
# TAB 4: EXPLORE DATA

with tab_data:

    st.subheader("Dataset Exploration")

    # Dataset Preview
    st.write("### Dataset Preview")

    st.dataframe(
        df.head(30),
        use_container_width=True
    )

    # Summary Statistics
    st.write("### Summary Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    # Missing Values
    st.write("### Missing Values")

    missing_values = (
        df.isnull().sum()
        .rename("Missing Values")
        .to_frame()
    )

    st.dataframe(
        missing_values,
        use_container_width=True
    )

    # Correlation Heatmap
    st.write("### Feature Correlation Matrix")

    numeric_data = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Feature Correlation Heatmap")

    st.pyplot(fig)

    plt.close(fig)
# TAB 5: SAVE MODEL

with tab_download:

    st.subheader("Download Trained Machine Learning Model")

    st.write(
        "Select the machine learning algorithm "
        "whose trained model you want to download."
    )

    model_to_save = st.selectbox(
        "Select Model",
        list(models.keys()),
        key="export_model"
    )

    import io

    buffer = io.BytesIO()

    joblib.dump(
        models[model_to_save],
        buffer
    )

    st.download_button(
        label="Download Trained Model (.pkl)",
        data=buffer.getvalue(),
        file_name="heart_disease_model.pkl",
        mime="application/octet-stream"
    )

    st.success("Your selected model is ready for download.")

    st.caption(
        "The downloaded pipeline includes its fitted "
        "preprocessing and classifier."
    )