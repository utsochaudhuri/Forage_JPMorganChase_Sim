import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# Load your data (as you had it)
df = pd.read_csv("Task 3/Task 3 and 4_Loan_Data.csv", index_col="customer_id")

# ----- Features -----
feature_cols = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
    "dti",
    "loan_to_income",
]

def make_features(dfin: pd.DataFrame) -> pd.DataFrame:
    f = dfin.copy()
    eps = 1e-6
    denom = np.maximum(f["income"].values, eps)
    f["dti"] = f["total_debt_outstanding"].values / denom
    f["loan_to_income"] = f["loan_amt_outstanding"].values / denom
    X = f[feature_cols]
    return X

# Target and X
y = df["default"].astype(int)
X = make_features(df).replace([np.inf, -np.inf], np.nan)
valid_idx = X.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LPM
lpm = LinearRegression().fit(X_tr, y_tr)
y_pred = np.clip(lpm.predict(X_te), 0, 1)

# Predictors
def predict_pd_lpm_by_id(customer_id: int) -> float:
    x = make_features(df.loc[[customer_id]])
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # quick guard
    return float(np.clip(lpm.predict(x)[0], 0, 1))

def predict_pd_lpm(person: dict) -> float:
    x = make_features(pd.DataFrame([person]))
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(np.clip(lpm.predict(x)[0], 0, 1))

def exp_loss(pd_val, loan_amount):
    return pd_val*0.9*loan_amount

# Example usage
pd_val = predict_pd_lpm({
    "credit_lines_outstanding": 1,
    "loan_amt_outstanding": 12000,
    "total_debt_outstanding": 25000,
    "income": 60000,
    "years_employed": 3,
    "fico_score": 690
})
loan_amount = 12000
expected_loss = exp_loss(pd_val, loan_amount)
print(f"Expected Loss: {expected_loss:.2f}")