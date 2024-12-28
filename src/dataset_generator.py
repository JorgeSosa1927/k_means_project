from sklearn.datasets import load_iris
import pandas as pd
import os

# Ensure the 'benchmarks' directory exists
os.makedirs('../benchmarks', exist_ok=True)

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Scale the dataset to 70,000 rows by repeating it
scaled_data = pd.DataFrame(data).sample(n=70000, replace=True, random_state=42)

# Save to CSV
scaled_data.to_csv('../benchmarks/dataset.csv', index=False)

print("Dataset scaled to 70,000 samples and saved to 'benchmarks/dataset.csv'")
