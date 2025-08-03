import pandas as pd
from model import train_model 
from sklearn.metrics import roc_auc_score
from evaluate_model import evaluate_model
import mlflow

baseline = pd.read_csv("data/month_01.csv")
baseline_amt = baseline["amt"].mean()
baseline_pop = baseline["city_pop"].mean()
baseline_gender_dist = baseline["gender"].value_counts(normalize=True)
baseline_job_set = set(baseline["job"].dropna().unique())

mlflow.set_experiment("drift_detection")

for month in range(1, 13):
    filename = f"data/month_{month:02d}.csv"
    df = pd.read_csv(filename)

    current_amt = df["amt"].mean()
    current_pop = df["city_pop"].mean()
    current_gender_dist = df["gender"].value_counts(normalize=True)
    current_job_set = set(df["job"].dropna().unique())

    amt_ratio = current_amt / baseline_amt
    pop_ratio = current_pop / baseline_pop

    # Gender drift
    male_ratio = current_gender_dist.get("M", 0.5)  # fallback: 50/50
    baseline_male = baseline_gender_dist.get("M", 0.5)
    gender_diff = abs(male_ratio - baseline_male)

    # Job overlap
    job_overlap = len(baseline_job_set & current_job_set) / max(len(baseline_job_set), 1)

    print(f"\nMonth {month:02d}")
    print(f"  ➤ AMT Δ: {amt_ratio:.2f}, POP Δ: {pop_ratio:.2f}")
    print(f"  ➤ GENDER Δ(M): {gender_diff:.2f}, JOB Overlap: {job_overlap:.2f}")

    # Performace test for new month
    try:
        auc = evaluate_model(df)
        print(f"ROC AUC on month_{month:02d}: {auc:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        auc = 0  

    retrain = False
    if amt_ratio < 0.8 or amt_ratio > 1.2:
        retrain = True
        print("  Drift in AMT detected.")
    if pop_ratio < 0.85 or pop_ratio > 1.15:
        retrain = True
        print("  Drift in City Population detected.")
    if gender_diff > 0.1:
        retrain = True
        print("  Gender distribution changed.")
    if job_overlap < 0.7:
        retrain = True
        print("  Job category distribution changed.")
    if auc < 0.90:
        retrain = True
        print("  Model performance degraded.")

    with mlflow.start_run():
        mlflow.log_metric("amt_ratio", amt_ratio)
        mlflow.log_metric("pop_ratio", pop_ratio)
        mlflow.log_metric("gender_diff", gender_diff)
        mlflow.log_metric("job_overlap", job_overlap)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("month", month)

    print(f"\nMonth {month:02d}")
    print(f"  ➤ AMT Δ: {amt_ratio:.2f}, POP Δ: {pop_ratio:.2f}")
    print(f"  ➤ GENDER Δ(M): {gender_diff:.2f}, JOB Overlap: {job_overlap:.2f}")
    print(f"  ROC AUC: {auc:.4f}")

    if retrain:
        print("Retraining triggered.")
        train_model(df)
    else:
        print("No significant drift or performance drop.")

