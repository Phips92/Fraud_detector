import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv("fraudTrain.csv")

# Display the first few rows
print(df.head())

# Summary statistics
print(df.describe())

# Data types and missing values
print(df.info())

# Count and visualize the proportion of fraudulent vs. legitimate transactions
fraud_counts = df["is_fraud"].value_counts()
fraud_percent = fraud_counts / len(df) * 100

plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette="Set2")
plt.xticks([0, 1], ["Legit", "Fraud"])
for i, val in enumerate(fraud_percent):
    plt.text(i, fraud_counts[i] + 1000, f"{val:.2f}%", ha='center')
plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

# Violin plot of transaction amounts by fraud label (only for transactions under $5000)
df_violin = df[df["amt"] < 5000]

plt.figure(figsize=(8, 5))
sns.violinplot(data=df_violin, x="is_fraud", y="amt", palette="Pastel1", inner="quartile")
plt.title("Transaction Amount Distribution by Fraud Status (< $5000)")
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.ylabel("Transaction Amount $")
plt.show()

# Distribution plot (99th percentile filter), separated by fraud status
q99 = df['amt'].quantile(0.99)
df_filtered = df[df['amt'] < q99]
plt.figure(figsize=(12, 5))

# Legit
sns.histplot(df_filtered[df_filtered["is_fraud"] == 0]["amt"], bins=40, color="skyblue", label="Legit", kde=True, stat="density", alpha=0.6)

# Fraud
sns.histplot(df_filtered[df_filtered["is_fraud"] == 1]["amt"], bins=40, color="red", label="Fraud", kde=True, stat="density", alpha=0.6)

plt.title("Transaction Amount Distribution (Filtered < 99th Percentile)")
plt.xlabel("Amount")
plt.ylabel("Density")
plt.legend()
plt.show()

# Horizontal bar chart of the most frequent categories in fraud cases
plt.figure(figsize=(12, 6))
fraud_by_cat = df[df["is_fraud"] == 1]["category"].value_counts().head(10)
sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index, palette="Reds_r")
plt.title("Top 10 Fraudulent Categories")
plt.xlabel("Number of Fraudulent Transactions")
plt.ylabel("Category")
for i, v in enumerate(fraud_by_cat.values):
    plt.text(v + 5, i, str(v), va="center")
plt.show()

# Convert to datetime
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

# Extract day of the week
df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()

# Bar chart showing distribution of all transactions across weekdays
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="day_of_week", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], palette="Set3")
plt.title("Transactions by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Smoothed density plot of fraud occurrence by hour (shifted & reversed clock)
df["hour"] = df["trans_date_trans_time"].dt.hour
df["hour_shifted"] = (df["hour"] - 12) % 24
fraud_hours_shifted = df[df["is_fraud"] == 1]["hour_shifted"]

xticks = list(range(23, -1, -2))  
xtick_labels = [(i + 12) % 24 for i in xticks] 

plt.figure(figsize=(12, 5))
sns.kdeplot(fraud_hours_shifted, fill=True, color="crimson", linewidth=2)

plt.title("Smoothed Density of Fraudulent Transactions by Hour (Starting from Noon, Reversed)")
plt.xlabel("Hour of Day")
plt.ylabel("Density")

plt.xticks(ticks=xticks, labels=xtick_labels)
plt.gca().invert_xaxis()  
plt.tight_layout()
plt.show()

# Horizontal bar chart of the merchants with the highest transaction count
top_merchants = df["merchant"].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_merchants.values, y=top_merchants.index, palette="Blues_r")
plt.title("Top 10 Merchants by Transaction Count")
plt.xlabel("Number of Transactions")
plt.ylabel("Merchant")
plt.tight_layout()
plt.show()


# Histogram + KDE of ages involved in fraud cases
df["dob"] = pd.to_datetime(df["dob"])
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

fraud_ages = df[df["is_fraud"] == 1]["age"]

plt.figure(figsize=(10, 5))
sns.histplot(fraud_ages, bins=30, color="purple", kde=True)
plt.title("Age Distribution of Fraudulent Transactions")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()






