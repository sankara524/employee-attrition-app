import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"

# display name -> file name
MODELS = {
    "Random Forest": "Employee_Attrition_Random_Forest_Model.pkl",
    "Gradient Boosting": "Employee_Attrition_Gradient_Boosting_Model.pkl",
    "XGBoost": "Employee_Attrition_XGBoost_Model.pkl",
    "SVM": "Employee_Attrition_SVM_Model.pkl",
    "Logistic Regression": "Employee_Attrition_Logistic_Regression_Model.pkl",
    "KNN": "Employee_Attrition_KNN_Model.pkl",
    "Decision Tree": "Employee_Attrition_Decision_Tree_Model.pkl",
    "Naive Bayes": "Employee_Attrition_Naive_Bayes_Model.pkl",
}

# XGBoost and KNN were trained on numeric labels: 0 = Stayed, 1 = Left
NUMERIC_LABEL_MODELS = {"XGBoost", "KNN"}


@st.cache_resource(show_spinner="Loading model...")
def load_model(file_name: str):
    # XGBoost is stored as two portable files (preprocessor + booster JSON)
    # because the single-file pickle failed to load on some machines.
    if "XGBoost" in file_name:
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier

        pre = joblib.load(MODEL_DIR / "Employee_Attrition_XGBoost_preprocessor.pkl")
        clf = XGBClassifier()
        clf.load_model(str(MODEL_DIR / "Employee_Attrition_XGBoost_booster.json"))
        return Pipeline([("preprocessor", pre), ("xgb", clf)])
    return joblib.load(MODEL_DIR / file_name)


def predict(model, model_name: str, row: pd.DataFrame):
    """Return (label, probability_of_leaving) with label mapping handled."""
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
    classes = list(model.classes_)

    if model_name in NUMERIC_LABEL_MODELS:
        label = "Left" if int(pred) == 1 else "Stayed"
        p_left = float(proba[classes.index(1)]) if proba is not None else None
    else:
        label = str(pred)
        p_left = float(proba[classes.index("Left")]) if proba is not None else None
    return label, p_left


# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
if model_name in ("KNN", "SVM"):
    st.sidebar.info(f"{model_name} can take a few seconds to predict.")

# ---------------- Main form ----------------
st.title("👥 Employee Attrition Predictor")
st.write("Fill in the employee details, then click **Predict**.")

with st.form("employee_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Personal")
        age = st.number_input("Age", 18, 59, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Number of Dependents", 0, 6, 0)
        education = st.selectbox(
            "Education Level",
            ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"],
        )
        distance = st.number_input("Distance from Home", 1, 99, 25)

    with c2:
        st.subheader("Job")
        job_role = st.selectbox("Job Role", ["Education", "Finance", "Healthcare", "Media", "Technology"])
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        income = st.number_input("Monthly Income", 1000, 20000, 7000, step=100)
        years = st.number_input("Years at Company", 1, 51, 5)
        tenure = st.number_input("Company Tenure (months)", 2, 128, 55)
        promotions = st.number_input("Number of Promotions", 0, 4, 1)
        performance = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"])

    with c3:
        st.subheader("Work environment")
        wlb = st.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])
        satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
        recognition = st.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])
        reputation = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        remote = st.selectbox("Remote Work", ["No", "Yes"])
        leadership = st.selectbox("Leadership Opportunities", ["No", "Yes"])
        innovation = st.selectbox("Innovation Opportunities", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # Column names must match the training data exactly
    row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Years at Company": years,
        "Job Role": job_role,
        "Monthly Income": income,
        "Work-Life Balance": wlb,
        "Job Satisfaction": satisfaction,
        "Performance Rating": performance,
        "Number of Promotions": promotions,
        "Overtime": overtime,
        "Distance from Home": distance,
        "Education Level": education,
        "Marital Status": marital,
        "Number of Dependents": dependents,
        "Job Level": job_level,
        "Company Size": company_size,
        "Company Tenure": tenure,
        "Remote Work": remote,
        "Leadership Opportunities": leadership,
        "Innovation Opportunities": innovation,
        "Company Reputation": reputation,
        "Employee Recognition": recognition,
    }])

    try:
        model = load_model(MODELS[model_name])
        label, p_left = predict(model, model_name, row)

        st.divider()
        if label == "Left":
            st.error(f"⚠️ **{model_name}** predicts this employee is likely to **LEAVE**.")
        else:
            st.success(f"✅ **{model_name}** predicts this employee is likely to **STAY**.")

        if p_left is not None:
            st.metric("Probability of leaving", f"{p_left:.1%}")
            st.progress(min(max(p_left, 0.0), 1.0))
    except FileNotFoundError:
        st.error(f"Model file not found: models/{MODELS[model_name]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
