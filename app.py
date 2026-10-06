import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🩺 Diabetes Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)

    
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🩺 Diabetes Prediction App")

    st.write(
        "This application uses Logistic Regression "
        "to predict the possibility of diabetes."
    )

    st.markdown("---")

    st.subheader("📌 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Logistic Regression")

    with col2:
        st.metric("Problem", "Binary Classification")

    with col3:
        st.metric("Target", "Outcome")

    st.markdown("---")

    st.subheader("How it works")

    st.write("""
    1. Enter the patient's medical information.
    2. Click the Predict button.
    3. The trained Logistic Regression model processes the inputs.
    4. The application displays the prediction and probability.
    """)

    st.info(
        "0 = No Diabetes | 1 = Diabetes"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Diabetes Prediction")

    st.write(
        "Enter the patient's details below."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INPUT FIELDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=250.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=150.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree],
            "Age": [age]
        })

        # Arrange columns exactly as used during training
        input_data = input_data[features]

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(
            input_scaled
        )[0]

        no_diabetes_probability = probability[0] * 100
        diabetes_probability = probability[1] * 100

        st.markdown("---")

        st.subheader("📋 Prediction Result")

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Diabetes Detected"
            )

        else:

            st.success(
                "✅ No Diabetes"
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "No Diabetes",
                f"{no_diabetes_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Diabetes",
                f"{diabetes_probability:.2f}%"
            )

        st.progress(
            int(diabetes_probability)
        )

        # Show entered values
        st.subheader("Patient Information")

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance metrics of the trained Logistic Regression model."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # IMPORTANT:
    # Replace these values with YOUR actual model results.
    # --------------------------------------------------------

    accuracy = 0.78
    precision = 0.75
    recall = 0.70
    f1 = 0.72
    roc_auc = 0.83

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{roc_auc:.2%}"
        )

    st.markdown("---")

    st.subheader("Model Information")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.info(
        "Update the metric values above with the actual "
        "results generated from your Jupyter Notebook."
    )

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.write("""
    This project uses Logistic Regression for binary
    classification of diabetes outcomes.
    """)

    st.markdown("---")

    st.subheader("Machine Learning Algorithm")

    st.write("Logistic Regression")

    st.subheader("Input Features")

    st.write("""
    • Pregnancies

    • Glucose

    • Blood Pressure

    • Skin Thickness

    • Insulin

    • BMI

    • Diabetes Pedigree Function

    • Age
    """)

    st.subheader("Target Variable")

    st.write("""
    Outcome

    0 → No Diabetes

    1 → Diabetes
    """)

    st.markdown("---")

    st.warning(
        "This application is a machine-learning demonstration "
        "and should not be used as a medical diagnosis."
    )