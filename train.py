import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score, confusion_matrix
)
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/csv_Phishing_Dataset.csv")
print("Initial shape:", df.shape)
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Drop duplicate rows
dup_count = df.duplicated().sum()
print("Duplicate rows:", dup_count)
df = df.drop_duplicates()

# Quick checks
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print(df['Result'].value_counts().head())

# ---- Boxplots of top features ----
if "Result" in df.columns:
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    numeric_cols.remove("Result")
    for col in numeric_cols[:6]:  # first 6 features
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Result", y=col, data=df)
        plt.title(f"Feature vs Target: {col}")
        plt.show()

# ---- Histograms ----
df.hist(bins=30, figsize=(15,12))
plt.suptitle("Feature Distributions", size=16)
plt.show()

print("\n-- Descriptive stats (numeric) --")
print(df.describe().T)

plt.figure(figsize=(5,4))
sns.countplot(x="Result", data=df)
plt.title("Target Class Distribution (Result)")
plt.show()
plt.close()

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()
plt.close()

# The dataset uses -1/1; convert to 0/1 for ML
df["Result"] = df["Result"].replace({-1: 0, 1: 1})
y = df["Result"]
X = df.drop("Result", axis=1)

# Keep column order (we'll use this later)
all_features = X.columns.tolist()
print("Number of features:", len(all_features))

lexical_features = []
# pick commonly lexical features from your dataset if present
candidates = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
    "HTTPS_token", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
    "Abnormal_URL", "Redirect", "on_mouseover", "RightClick", "popUpWidnow",
    "Iframe"
]
for c in candidates:
    if c in X.columns:
        lexical_features.append(c)

print("Lexical features used for URL-based pipeline:", lexical_features)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Also make lexical train/test if lexical features present
if lexical_features:
    X_train_lex = X_train_full[lexical_features].copy()
    X_test_lex = X_test_full[lexical_features].copy()
else:
    X_train_lex = None
    X_test_lex = None

print("Train/test shapes (full):", X_train_full.shape, X_test_full.shape)
if lexical_features:
    print("Train/test shapes (lexical):", X_train_lex.shape, X_test_lex.shape)
def evaluate_and_print(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except Exception:
        pass
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print(f"\n{name} metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {roc if roc else 'n/a'}")
    print(classification_report(y_test, y_pred))
    return {"name": name, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "roc_auc": roc}
results = []

# Standard scaler + classifier in pipeline
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 7a) Logistic Regression (full)
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
pipe_lr.fit(X_train_full, y_train)
results.append(evaluate_and_print("LogisticRegression (full)", pipe_lr, X_test_full, y_test))
# 7b) SVM (full) -- enable probability for ROC
pipe_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=42))
])
pipe_svc.fit(X_train_full, y_train)
results.append(evaluate_and_print("SVC (full)", pipe_svc, X_test_full, y_test))
# 7c) Random Forest (full)
pipe_rf = Pipeline([
    ("scaler", StandardScaler()),   # RF doesn't need scaling, but keep pipeline consistent
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
pipe_rf.fit(X_train_full, y_train)
results.append(evaluate_and_print("RandomForest (full)", pipe_rf, X_test_full, y_test))
if lexical_features:
    # Logistic on lexical
    pipe_lr_lex = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    pipe_lr_lex.fit(X_train_lex, y_train)
    results.append(evaluate_and_print("LogisticRegression (lexical)", pipe_lr_lex, X_test_lex, y_test))

    # SVC on lexical
    pipe_svc_lex = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
    pipe_svc_lex.fit(X_train_lex, y_train)
    results.append(evaluate_and_print("SVC (lexical)", pipe_svc_lex, X_test_lex, y_test))

    # RandomForest on lexical
    pipe_rf_lex = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    pipe_rf_lex.fit(X_train_lex, y_train)
    results.append(evaluate_and_print("RandomForest (lexical)", pipe_rf_lex, X_test_lex, y_test))
full_results = [r for r in results if "(full)" in r['name']]
best_full = max(full_results, key=lambda x: x['f1'])
print("\nBest full-feature model:", best_full)

# choose best lexical model if available
lex_results = [r for r in results if "(lexical)" in r['name']]
best_lex = max(lex_results, key=lambda x: x['f1']) if lex_results else None
print("Best lexical model:", best_lex)
# Map names to actual objects to save pipelines properly
pipeline_map = {
    "LogisticRegression (full)": pipe_lr,
    "SVC (full)": pipe_svc,
    "RandomForest (full)": pipe_rf
}
if lexical_features:
    pipeline_map.update({
        "LogisticRegression (lexical)": pipe_lr_lex,
        "SVC (lexical)": pipe_svc_lex,
        "RandomForest (lexical)": pipe_rf_lex
    })
best_full_pipeline_name = best_full['name']
best_full_pipeline = pipeline_map[best_full_pipeline_name]
joblib.dump(best_full_pipeline, "best_full_pipeline.pkl")
print("Saved best full pipeline as best_full_pipeline.pkl ->", best_full_pipeline_name)
# Save the best lexical pipeline (for runtime URL -> lexical features)
if best_lex:
    best_lex_pipeline_name = best_lex['name']
    best_lex_pipeline = pipeline_map[best_lex_pipeline_name]
    joblib.dump(best_lex_pipeline, "best_lexical_pipeline.pkl")
    print("Saved best lexical pipeline as best_lexical_pipeline.pkl ->", best_lex_pipeline_name)
# Also save the list of features used in lexical pipeline to disk for the Streamlit app
if lexical_features:
    joblib.dump(lexical_features, "lexical_features_list.pkl")
    print("Saved lexical feature list as lexical_features_list.pkl")


# Save a CSV of top feature importances using the full RandomForest if available
try:
    # If full RF exists, get feature importance
    if isinstance(pipe_rf.named_steps['clf'], RandomForestClassifier):
        rf = pipe_rf.named_steps['clf']
        importances = rf.feature_importances_
        feat_imp = pd.Series(importances, index=X_train_full.columns).sort_values(ascending=False)
        feat_imp.head(25).to_csv("top25_feature_importances.csv")
        print("Saved top25_feature_importances.csv")
except Exception:
    pass

print("\nTraining & saving complete.")

