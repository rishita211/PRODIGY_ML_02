import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === STEP 1: Extract ZIP file ===
zip_path = r"C:\Users\RISHITA\Downloads\archive (1).zip"  # <--- zip path
extract_dir = "customer_data"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("[INFO] ZIP extracted!")

# === STEP 2: Find the CSV file ===
csv_file = None
for file in os.listdir(extract_dir):
    if file.endswith(".csv"):
        csv_file = os.path.join(extract_dir, file)
        break

if not csv_file:
    raise FileNotFoundError("❌ CSV file not found in the ZIP!")

df = pd.read_csv(csv_file)
print("[INFO] Data Loaded:")
print(df.head())

# === STEP 3: Preprocess features ===
# Use only numeric columns (remove ID columns if present)
features = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
feature_cols = [col for col in features.columns if 'id' not in col.lower()]
X = features[feature_cols]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === STEP 4: Apply KMeans ===
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# === STEP 5: Plot the clusters (first 2 features) ===
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='Set1', s=80)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Customer Segments')
plt.grid(True)
plt.tight_layout()

# === STEP 6: Save results ===
plot_path = "customer_clusters_plot.png"
csv_output_path = "customer_clusters_output.csv"

plt.savefig(plot_path)
df.to_csv(csv_output_path, index=False)

print(f"[✅] Clustered data saved to {csv_output_path}")
print(f"[✅] Plot saved to {plot_path}")
