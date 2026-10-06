import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Logistic Regression Prediction")

st.write("Enter the input values below")

# Example inputs
age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

income = st.number_input(
    "Income",
    min_value=0.0,
    value=50000.0
)

education = st.number_input(
    "Education",
    min_value=0.0,
    value=12.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Education': [education]
    })

    st.write("Input:")
    st.write(input_data)

    st.success("Prediction generated")