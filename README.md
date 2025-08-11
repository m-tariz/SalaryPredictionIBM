# Salary Prediction using Ensemble Learning

This is an IBM Project Based Learning (PBL) project to build a **Salary Prediction** model using **Ensemble Learning** (Random Forest).  
The app predicts salary based on Age, Gender, Education Level, Job Title, and Years of Experience.

---

## Project Overview

The goal is to create a reliable ML model that accurately predicts salary using key features.  
A Streamlit web app provides an interactive UI to input details and get salary predictions in real-time.

---

## Dataset

- Contains: `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`, `Salary`  
- Data is loaded directly from a Google Drive link within the app code. No manual download needed.

---

## Features

- Handles categorical data with OneHotEncoding  
- Uses Random Forest Regressor (Ensemble Learning) for better accuracy  
- Interactive Streamlit app with dropdowns and input fields  
- Instant salary prediction and display

---

## Requirements

- Python 3.7+  
- Dependencies in `requirements.txt` (streamlit, scikit-learn, pandas, numpy)

Install packages with:

```bash
pip install -r requirements.txt
