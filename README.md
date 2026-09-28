# Employee Attrition Prediction System

Streamlit app that predicts whether an employee will (Stay/Leave), using 8 trained machine learning models.

## Folder layout
```
Employee-Attrition-app/
├── app.py
├── requirements.txt
└── models/
    ├── Employee_Attrition_Random_Forest_Model.pkl
    ├── Employee_Attrition_Gradient_Boosting_Model.pkl
    ├── Employee_Attrition_XGBoost_Model.pkl
    ├── Employee_Attrition_SVM_Model.pkl
    ├── Employee_Attrition_Logistic_Regression_Model.pkl
    ├── Employee_Attrition_KNN_Model.pkl
    ├── Employee_Attrition_Decision_Tree_Model.pkl
    └── Employee_Attrition_Naive_Bayes_Model.pkl
```
## Tech Stack
Python, Pandas, scikit-learn, Streamlit

# Dataset
Employee Attrition dataset(59,598 records, 24 columns).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
