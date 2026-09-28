import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"

# display name -> file name
MODELS = {
    "Random Forest": "Employee_Attrition_Random_Forest_Model.pkl",
    "Gradient Boosting": "Employee_Attrition_Gradient_Boosting_Model.pkl",
    "XGBoost": "Employee_Attrition_XGBoost_Model.pkl",  # actually loaded from preprocessor .pkl + booster .json (see load_model)
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


CATEGORICAL = [
    "Gender", "Job Role", "Work-Life Balance", "Job Satisfaction", "Performance Rating",
    "Overtime", "Education Level", "Marital Status", "Job Level", "Company Size",
    "Remote Work", "Leadership Opportunities", "Innovation Opportunities",
    "Company Reputation", "Employee Recognition",
]


def compare_all_models(row: pd.DataFrame) -> pd.DataFrame:
    """Run every model on the same input and collect prediction + probability."""
    results = []
    for name, file_name in MODELS.items():
        try:
            label, p_left = predict(load_model(file_name), name, row)
            results.append({"Model": name, "Prediction": label, "Probability of leaving": p_left})
        except Exception as e:  # keep going if one model fails
            results.append({"Model": name, "Prediction": f"error: {e}", "Probability of leaving": None})
    return pd.DataFrame(results)


@st.cache_data(show_spinner="Calculating feature importance...")
def get_feature_importance() -> pd.Series:
    """Random Forest importance, with one-hot columns summed back to the original feature."""
    pipe = load_model(MODELS["Random Forest"])
    pre, rf = pipe.steps[0][1], pipe.steps[-1][1]
    totals = {}
    for col_name, value in zip(pre.get_feature_names_out(), rf.feature_importances_):
        base = col_name.split("__", 1)[1]
        original = next(
            (c for c in sorted(CATEGORICAL, key=len, reverse=True) if base.startswith(c + "_")),
            base,
        )
        totals[original] = totals.get(original, 0.0) + float(value)
    return pd.Series(totals).sort_values(ascending=False)


def probability_chart(valid: pd.DataFrame) -> alt.Chart:
    """Horizontal bars sorted by probability, fixed 0-100% axis, red = Leave, green = Stay."""
    data = valid[["Model", "Probability of leaving"]].rename(columns={"Probability of leaving": "Probability"})
    data["Outcome"] = data["Probability"].apply(lambda p: "Leave" if p >= 0.5 else "Stay")

    x = alt.X(
        "Probability:Q",
        scale=alt.Scale(domain=[0, 1.12]),  # a little room so the labels are never clipped
        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1.0], title="Probability of leaving"),
    )
    order = data.sort_values("Probability", ascending=False)["Model"].tolist()
    y = alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))

    bars = alt.Chart(data).mark_bar().encode(
        x=x,
        y=y,
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(domain=["Leave", "Stay"], range=["#ff4b4b", "#21c354"]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=["Model", alt.Tooltip("Probability:Q", format=".1%"), "Outcome"],
    )
    labels = alt.Chart(data).mark_text(align="left", dx=5, color="#fafafa").encode(
        x=alt.X("Probability:Q"), y=y, text=alt.Text("Probability:Q", format=".1%")
    )
    threshold = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        strokeDash=[5, 5], color="#fafafa", opacity=0.6
    ).encode(x=alt.X("x:Q"))
    return (bars + labels + threshold).properties(height=alt.Step(34))


def importance_chart(top: pd.Series) -> alt.Chart:
    """Horizontal bars sorted from most to least important, shown as a share of the total."""
    data = top.rename("Importance").rename_axis("Feature").reset_index()
    order = data.sort_values("Importance", ascending=False)["Feature"].tolist()
    y = alt.Y("Feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    bars = alt.Chart(data).mark_bar(color="#4c9be8").encode(
        x=alt.X("Importance:Q", axis=alt.Axis(format="%", title="Share of total importance")),
        y=y,
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
    )
    labels = alt.Chart(data).mark_text(align="left", dx=5, color="#fafafa").encode(
        x=alt.X("Importance:Q"), y=y, text=alt.Text("Importance:Q", format=".1%")
    )
    return (bars + labels).properties(height=alt.Step(30))


# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
compare = st.sidebar.checkbox("Also compare all 8 models", value=True)
if model_name in ("KNN", "SVM") or compare:
    st.sidebar.info("KNN and SVM can take a few seconds the first time.")

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

        if compare:
            st.divider()
            st.subheader("📊 All models compared")
            with st.spinner("Running all 8 models..."):
                table = compare_all_models(row)

            valid = table.dropna(subset=["Probability of leaving"])
            n_leave = int((valid["Prediction"] == "Left").sum())
            m1, m2 = st.columns(2)
            m1.metric("Models predicting LEAVE", f"{n_leave} of {len(valid)}")
            m2.metric("Average probability of leaving", f"{valid['Probability of leaving'].mean():.1%}")

            if n_leave > len(valid) / 2:
                st.error("Majority vote: this employee is likely to **LEAVE**.")
            elif n_leave < len(valid) / 2:
                st.success("Majority vote: this employee is likely to **STAY**.")
            else:
                st.warning("The models are split evenly on this employee.")

            st.dataframe(
                table.assign(**{"Probability of leaving": table["Probability of leaving"].map(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")}),
                hide_index=True,
                width="stretch",
            )
            st.altair_chart(probability_chart(valid), width="stretch")
            st.caption("Dashed line = 50% decision threshold. Red bars predict Leave, green bars predict Stay.")
    except FileNotFoundError as e:
        st.error(f"Model file not found in models/ folder: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- Feature importance ----------------
st.divider()
with st.expander("🔍 What drives employee attrition? (feature importance)"):
    st.write(
        "Which employee and workplace factors matter most in the Random Forest model "
        "across the whole dataset. Higher means more influence on the prediction. "
        "This is a general view, not an explanation of the single employee above."
    )
    try:
        importance = get_feature_importance()
        top = importance.head(10).rename("Importance")
        st.altair_chart(importance_chart(top), width="stretch")
    except Exception as e:
        st.info(f"Feature importance is not available: {e}")