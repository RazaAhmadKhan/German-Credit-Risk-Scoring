"""
CodeAlpha_CreditScoringModel — Task 1: Credit Scoring Model
============================================================================
Predicts an individual's creditworthiness (Good / Bad credit risk) using the
Statlog German Credit Dataset (UCI Machine Learning Repository).

This script:
  1. Downloads the dataset directly from a public URL (no manual download needed)
  2. Encodes categorical financial/personal history features
  3. Trains & compares multiple classification algorithms
  4. Evaluates each with Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  5. Selects the best model automatically (by ROC-AUC) and saves the full
     preprocessing + model pipeline to disk for use by the GUI (app.py)

Author: Raza Ahmad Khan
Repo:   CodeAlpha_CreditScoringModel
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATASET DIRECTLY FROM URL
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"

COLUMN_NAMES = [
    "CheckingStatus", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatus", "OtherDebtors",
    "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
    "ExistingCredits", "Job", "NumLiable", "Telephone", "ForeignWorker", "Target"
]

print("=" * 70)
print("STEP 1: Downloading Statlog German Credit Dataset from UCI (via URL)")
print("=" * 70)
df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
print(f"Dataset shape: {df.shape}")
print(df.head(), "\n")

# Target: 1 = Good credit risk, 2 = Bad credit risk -> convert to 1 = Good, 0 = Bad
df["Target"] = df["Target"].map({1: 1, 2: 0})
print("Class balance:\n", df["Target"].value_counts(), "\n")

# ---------------------------------------------------------------------------
# 2. FEATURE / TARGET SPLIT
# ---------------------------------------------------------------------------
X = df.drop("Target", axis=1)
y = df["Target"]

numeric_features = ["Duration", "CreditAmount", "InstallmentRate",
                     "ResidenceSince", "Age", "ExistingCredits", "NumLiable"]
categorical_features = [c for c in X.columns if c not in numeric_features]

print("=" * 70)
print("STEP 2: Feature setup")
print("=" * 70)
print(f"Numeric features    : {numeric_features}")
print(f"Categorical features: {categorical_features}\n")

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

print("=" * 70)
print(f"STEP 3: Train/Test split -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("=" * 70, "\n")

# ---------------------------------------------------------------------------
# 4. TRAIN & COMPARE MULTIPLE MODELS (each wrapped with the preprocessor)
# ---------------------------------------------------------------------------
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
}
if XGB_AVAILABLE:
    base_models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE
    )

print("=" * 70)
print("STEP 4: Training & evaluating models")
print("=" * 70)

results = []
trained_pipelines = {}

for name, clf in base_models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()

    trained_pipelines[name] = pipe
    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc, "CV ROC-AUC": cv_score
    })

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}  (5-fold CV: {cv_score:.4f})")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("STEP 5: Model comparison summary (ranked by ROC-AUC)")
print("=" * 70)
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. SELECT BEST MODEL & SAVE (full pipeline: preprocessing + model in one file)
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
best_metrics = results_df.iloc[0].to_dict()

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}  (ROC-AUC = {best_metrics['ROC-AUC']:.4f})")
print("=" * 70)
print(classification_report(y_test, best_pipeline.predict(X_test)))

joblib.dump(best_pipeline, "credit_model_pipeline.pkl")

# Human-readable label maps for the GUI (code -> friendly description)
category_labels = {
    "CheckingStatus": {
        "A11": "< 0 DM (overdrawn)", "A12": "0 to 200 DM", "A13": ">= 200 DM", "A14": "No checking account"
    },
    "CreditHistory": {
        "A30": "No credits taken", "A31": "All credits paid back duly (this bank)",
        "A32": "Existing credits paid back duly till now", "A33": "Delay in past payments",
        "A34": "Critical account / other credits existing"
    },
    "Purpose": {
        "A40": "New car", "A41": "Used car", "A42": "Furniture/equipment",
        "A43": "Radio/television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other"
    },
    "Savings": {
        "A61": "< 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
        "A64": ">= 1000 DM", "A65": "Unknown / no savings account"
    },
    "Employment": {
        "A71": "Unemployed", "A72": "< 1 year", "A73": "1 to 4 years",
        "A74": "4 to 7 years", "A75": ">= 7 years"
    },
    "PersonalStatus": {
        "A91": "Male: divorced/separated", "A92": "Female: divorced/separated/married",
        "A93": "Male: single", "A94": "Male: married/widowed", "A95": "Female: single"
    },
    "OtherDebtors": {
        "A101": "None", "A102": "Co-applicant", "A103": "Guarantor"
    },
    "Property": {
        "A121": "Real estate", "A122": "Building society savings / life insurance",
        "A123": "Car or other property", "A124": "Unknown / no property"
    },
    "OtherInstallmentPlans": {
        "A141": "Bank", "A142": "Stores", "A143": "None"
    },
    "Housing": {
        "A151": "Rent", "A152": "Own", "A153": "For free"
    },
    "Job": {
        "A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
        "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified"
    },
    "Telephone": {
        "A191": "None", "A192": "Registered under customer's name"
    },
    "ForeignWorker": {
        "A201": "Yes", "A202": "No"
    },
}

metadata = {
    "best_model_name": best_model_name,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "category_labels": category_labels,
    "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in best_metrics.items() if k != "Model"},
    "numeric_ranges": {
        col: {"min": float(X[col].min()), "max": float(X[col].max()), "median": float(X[col].median())}
        for col in numeric_features
    },
}
with open("credit_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

results_df.to_csv("credit_model_comparison_results.csv", index=False)

print("\nSaved: credit_model_pipeline.pkl, credit_model_metadata.json, credit_model_comparison_results.csv")
print("Ready to run the GUI: streamlit run app.py")
