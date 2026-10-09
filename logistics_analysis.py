"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
Run: python logistics_analysis.py
Dependencies: pandas, numpy, matplotlib
Dataset: logistics_dataset.csv (hypothetical/simulated)
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_dataset.csv")

numeric = ["Distance_km", "Shipment_Volume_units", "Delivery_Time_days", "Transport_Cost_INR"]
print("Dataset shape:", df.shape)
print("\nDescriptive statistics:\n", df[numeric].describe().round(2))
print("\nCorrelation matrix:\n", df[numeric].corr().round(2))
print("\nBy transport mode:\n", df.groupby("Transport_Mode").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delivery_Days=("Delivery_Time_days", "mean"),
    Avg_Cost_INR=("Transport_Cost_INR", "mean"),
    On_Time_Rate=("On_Time", lambda x: (x == "Yes").mean() * 100)
).round(2))
print("\nBy region:\n", df.groupby("Region").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delivery_Days=("Delivery_Time_days", "mean"),
    Avg_Cost_INR=("Transport_Cost_INR", "mean")
).round(2))

plt.figure(figsize=(8, 4.5))
plt.hist(df["Delivery_Time_days"], bins=14, edgecolor="black")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of shipments")
plt.tight_layout()
plt.savefig("delivery_time_distribution.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.5))
df.groupby("Transport_Mode")["Delivery_Time_days"].mean().sort_values().plot(kind="bar")
plt.title("Average Delivery Time by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("Average delivery time (days)")
plt.tight_layout()
plt.savefig("avg_delivery_by_mode.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.5))
plt.scatter(df["Distance_km"], df["Transport_Cost_INR"], alpha=0.75)
plt.title("Distance vs Transport Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transport cost (INR)")
plt.tight_layout()
plt.savefig("distance_vs_cost.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.5))
df.groupby("Region")["Transport_Cost_INR"].mean().sort_values().plot(kind="bar")
plt.title("Average Transport Cost by Region")
plt.xlabel("Region")
plt.ylabel("Average cost (INR)")
plt.tight_layout()
plt.savefig("avg_cost_by_region.png", dpi=160)
plt.show()

corr = df[numeric].corr()
plt.figure(figsize=(7, 5))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=160)
plt.show()
