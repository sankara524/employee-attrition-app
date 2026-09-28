# 👥 Employee Attrition Prediction System

A Streamlit web app that predicts whether an employee is likely to **leave or stay**, using 8 machine learning models trained on an employee attrition dataset.

### 🚀 [Live demo](https://employee-attrition-app-nhia9bg3qjdxv6c5kax4ek.streamlit.app)

> The app runs on Streamlit Community Cloud's free tier. If it has been idle, click **"Wake up"** and wait a few seconds.

---

## Features

- **Predict** attrition for one employee from 22 workplace and personal features
- **Choose from 8 models**: Random Forest, Gradient Boosting, XGBoost, SVM, Logistic Regression, KNN, Decision Tree, Naive Bayes
- **Compare all models at once** on the same employee, with a table, a sorted probability chart and a majority vote
- **Feature importance** view showing which factors matter most in the Random Forest model
- Shows the **probability of leaving**, not just a Stay/Leave label

## Dataset

Employee Attrition dataset with **59,598 records** and 24 columns (Stayed 52.5%, Left 47.5%). The models use 22 features:

| Type | Features |
|---|---|
| Numeric | Age, Years at Company, Monthly Income, Number of Promotions, Distance from Home, Number of Dependents, Company Tenure |
| Categorical | Gender, Job Role, Work-Life Balance, Job Satisfaction, Performance Rating, Overtime, Education Level, Marital Status, Job Level, Company Size, Remote Work, Leadership Opportunities, Innovation Opportunities, Company Reputation, Employee Recognition |

Each model is a scikit-learn `Pipeline` that includes its own preprocessing (scaling and one-hot encoding), so the app passes raw values straight in.

## Tech stack

Python 3.12 · Streamlit · scikit-learn 1.6.1 · XGBoost · pandas · NumPy · Altair

## Project structure

```
employee-attrition-app/
├── app.py                 # Streamlit app
├── requirements.txt       # Pinned dependencies
├── models/                # Trained models (.pkl, plus XGBoost booster .json)
└── notebooks/             # EDA and model training notebooks
```

## Run locally

Python 3.12 is recommended.

```bash
git clone https://github.com/sankara524/employee-attrition-app.git
cd employee-attrition-app

python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Notes

- **Feature importance** is a general view of the Random Forest model across the dataset. It shows what the model relies on, not proven causes of attrition.
- Predictions are estimates from historical data and should support, not replace, human judgment.