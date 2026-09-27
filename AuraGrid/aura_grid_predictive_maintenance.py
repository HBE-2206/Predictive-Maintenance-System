"""AuraGrid Systems — Predictive Maintenance & Failure Classification."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "AuraGrid_Manufacturing_Operations.csv"
FEATURES = ["Factory_Line", "Operation_Type", "Total_Energy_kWh", "Production_Units", "Machine_Availability_Pct"]
CATEGORICAL = ["Factory_Line", "Operation_Type"]
NUMERICAL = ["Total_Energy_kWh", "Production_Units", "Machine_Availability_Pct"]
df = pd.read_csv(DATA_PATH)

pre_linear = ColumnTransformer([("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL), ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERICAL)])
pre_tree = ColumnTransformer([("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL), ("num", SimpleImputer(strategy="median"), NUMERICAL)])

# Classification: exclude rows with missing target labels.
cls = df.dropna(subset=["Failure_Type"])
X, y = cls[FEATURES], cls["Failure_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
models_cls = {
    "Logistic Regression": Pipeline([("prep", pre_linear), ("model", LogisticRegression(max_iter=2000))]),
    "Random Forest": Pipeline([("prep", pre_tree), ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
}
for name, model in models_cls.items():
    model.fit(X_train, y_train); pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Weighted F1:", round(f1_score(y_test, pred, average="weighted", zero_division=0), 4))
    print("Macro F1:", round(f1_score(y_test, pred, average="macro", zero_division=0), 4))
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred, labels=sorted(y.unique())))

# Regression: Repair_Cost_USD.
X, y = df[FEATURES], df["Repair_Cost_USD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
models_reg = {
    "Linear Regression": Pipeline([("prep", pre_linear), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("prep", pre_tree), ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))]),
}
for name, model in models_reg.items():
    model.fit(X_train, y_train); pred = model.predict(X_test)
    print(f"\n{name}")
    print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 2))
    print("MAE:", round(mean_absolute_error(y_test, pred), 2))
    print("R2:", round(r2_score(y_test, pred), 4))
