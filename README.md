# Credit Scoring Model 💳

**Machine Learning Internship — CodeAlpha | Task 1: Credit Scoring Model**

A machine learning system that predicts an individual's **creditworthiness** (Good vs Bad credit risk) from financial and personal history, with a real-time interactive web GUI for live predictions on unseen applicants.

## 📊 Overview

This project uses the **Statlog German Credit Dataset** (UCI Machine Learning Repository) to train and compare multiple classification algorithms, automatically selects the best-performing model, and serves it through a **Streamlit** web interface for real-time predictions.

- **Dataset**: Auto-downloaded from a public URL — no manual download needed
- **Algorithms compared**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Best model selected automatically** by 5-fold cross-validated ROC-AUC
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **GUI**: Real-time prediction on new/unseen applicant data via a web form with friendly dropdown labels

## 🗂 Project Structure

```
CodeAlpha_CreditScoringModel/
├── train_model.py                      # Downloads data, trains & evaluates models, saves best pipeline
├── app.py                              # Streamlit GUI for real-time predictions
├── requirements.txt                    # Python dependencies
├── credit_model_pipeline.pkl           # Saved best model + preprocessing pipeline (generated)
├── credit_model_metadata.json          # Best model info, metrics, label mappings (generated)
├── credit_model_comparison_results.csv # Full comparison table of all models (generated)
└── README.md
```

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This downloads the Statlog German Credit dataset directly from a public URL, encodes categorical features, trains 4 classification models, prints a full metrics comparison, and saves the best model pipeline to disk.

### 3. Launch the real-time prediction GUI
```bash
streamlit run app.py
```
Restart it fresh each time after retraining (don't just refresh the browser) so it picks up the latest model.

This opens a browser window where you can enter an applicant's financial history (checking account status, credit history, loan purpose, amount, employment, etc.) and get an instant credit risk prediction with probability score.

## 📈 Results

The training script evaluates every model using Precision, Recall, F1-Score, and ROC-AUC (with 5-fold cross-validation), then automatically picks the model with the highest ROC-AUC. Full results are printed to console and saved in `credit_model_comparison_results.csv`.

## 🧠 Approach

1. **Feature engineering** — 20 financial/personal attributes (7 numeric, 13 categorical) covering checking account status, credit history, savings, employment, property, and more.
2. **Preprocessing pipeline** — `StandardScaler` for numeric features + `OneHotEncoder` for categorical features, combined via `ColumnTransformer`, packaged in a single `sklearn.Pipeline` alongside the classifier.
3. **Model comparison** — Multiple classifiers (with class balancing, since the dataset has ~70/30 good/bad class split) trained and evaluated on a held-out test set.
4. **Model selection** — Best model chosen by ROC-AUC, then saved as one complete pipeline with `joblib` (no separate encoder/scaler files needed).
5. **Deployment** — Streamlit GUI loads the saved pipeline and presents human-readable dropdowns (e.g. "Checking Account Status: < 0 DM (overdrawn)") that map internally to the dataset's original coded values.

## ⚠️ Disclaimer

This tool is for educational/demonstration purposes only and does **not** constitute a real lending or financial decision. Always consult a certified financial advisor or loan officer for actual credit decisions.

## 👤 Author

**Raza Ahmad Khan**
CS & AI Student, UET Mardan
CodeAlpha Machine Learning Intern
