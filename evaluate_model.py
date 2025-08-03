from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_model(df):


    df = df.copy()
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
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, y_proba)

