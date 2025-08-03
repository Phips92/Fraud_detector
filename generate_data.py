import pandas as pd
import numpy as np
import random
import os

df = pd.read_csv("fraudTest.csv")
os.makedirs("data", exist_ok=True)

np.random.seed(42)

job_pool = df["job"].dropna().unique()

for i in range(1, 13):
    df_month = df.copy()
    
    amt_factor = 1 + (i - 6) * 0.05 + np.random.normal(0, 0.01)
    df_month["amt"] *= amt_factor

    # Simulate population drift (e.g., migration, urban growth)
    pop_drift = 1 + np.sin(i / 2) * 0.1
    df_month["city_pop"] = (df_month["city_pop"] * pop_drift).astype(int)

    # Slight geolocation drift for merchants
    df_month["merch_lat"] += np.random.normal(0, 0.02, size=len(df_month))
    df_month["merch_long"] += np.random.normal(0, 0.02, size=len(df_month))

    # Replace 5% of jobs with random alternatives
    mask = np.random.rand(len(df_month)) < 0.05
    df_month.loc[mask, "job"] = np.random.choice(job_pool, size=mask.sum())

    # Flip gender label in 3% of records (data noise)
    mask = np.random.rand(len(df_month)) < 0.03
    df_month.loc[mask, "gender"] = df_month.loc[mask, "gender"].map({"F": "M", "M": "F"})

    # Shift all transactions forward in time by i days
    df_month["trans_date_trans_time"] = pd.to_datetime(df_month["trans_date_trans_time"]) + pd.Timedelta(days=i)

    df_month.to_csv(f"data/month_{i:02d}.csv", index=False)

    print(f"Generated data/month_{i:02d}.csv with amt factor {amt_factor:.2f} and pop drift {pop_drift:.2f}")

