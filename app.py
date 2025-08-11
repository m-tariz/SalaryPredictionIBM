import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

@st.cache_data
def load_and_train():
    file_id = '1S3PvhwvdAZpadNZlUGGyJY_811T39G12'
    csv_url = f'https://drive.google.com/uc?id={file_id}'
    df = pd.read_csv(csv_url)
    df = df.dropna()

    X = df.drop('Salary', axis=1)
    y = df['Salary']

    categorical_features = ['Gender', 'Education Level', 'Job Title']
    numeric_features = ['Age', 'Years of Experience']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
    ])

    model.fit(X, y)
    return model

model = load_and_train()

st.title("Salary Prediction Web App")

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Select Gender", options=["Male", "Female", "Other"])
education = st.selectbox("Select Education Level", options=["Bachelors", "Masters", "PhD", "Other"])
job_title = st.text_input("Enter Job Title", value="Data Scientist")
years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [years_exp]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${prediction:,.2f}")
