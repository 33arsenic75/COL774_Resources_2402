import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("e.txt")

# Process the 'Hidden Layer' column
df["Hidden Layer List"] = df["Hidden Layer"].apply(lambda x: eval(x))
df["Depth"] = df["Hidden Layer List"].apply(len)
df["Architecture"] = df["Hidden Layer List"].apply(lambda x: str(x))


# Pivot for Train and Test F1
df_pivot = df.pivot(index="Architecture", columns="Dataset", values="F1").reset_index()

# Plot
plt.figure(figsize=(12, 6))
x = range(len(df_pivot))

plt.plot(x, df_pivot["Train"], marker='o', label="Train F1")
plt.plot(x, df_pivot["Test"], marker='o', label="Test F1")

# X-axis: architectures in shallower-to-deeper order
plt.xticks(x, df_pivot["Architecture"], rotation=45)

plt.xlabel("Architecture (Shallow to Deep)")
plt.ylabel("F1 Score")
plt.title("Train vs Test F1 Score vs Network Architecture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("E_Train_Test_F1_vs_depth.png")

