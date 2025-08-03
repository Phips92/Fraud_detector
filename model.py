from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.models.signature
from mlflow.models import infer_signature
import os

def train_model(df):

    df["dob"] = pd.to_datetime(df["dob"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    df = pd.get_dummies(df, columns=["merchant"], drop_first=True)
    scaler = StandardScaler()
    df["transaction_amount_scaled"] = scaler.fit_transform(df[["amt"]])


    merchant_columns = [col for col in df.columns if col.startswith("merchant_")]
    features = ["transaction_amount_scaled", "city_pop", "merch_lat", "merch_long", "age"] + merchant_columns

    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_model = X_train[features]
    X_test_model = X_test[features]


    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_model, y_train)

        y_pred = model.predict(X_test_model)
        y_proba = model.predict_proba(X_test_model)[:, 1]
        roc = roc_auc_score(y_test, y_proba)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc:.4f}")

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(model, "model")

        input_example = X_test_model.iloc[[0]]
        signature = infer_signature(X_test_model, model.predict_proba(X_test_model))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example, signature=signature)

        joblib.dump(model, "fraud_model.joblib")
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(merchant_columns, "merchant_columns.joblib")
