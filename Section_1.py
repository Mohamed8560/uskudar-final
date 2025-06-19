# section1_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("HIGGS.csv")

# Assume the first column is the target
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessed data for later sections
pd.DataFrame(X_scaled).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y).to_csv("y.csv", index=False)
