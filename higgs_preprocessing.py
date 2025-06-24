# higgs_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load a random 100,000 samples
def load_sample_data(filepath, n_samples=100000, random_state=42):
    column_names = ['label'] + [f'feature_{i}' for i in range(1, 29)]
    data = pd.read_csv(filepath, names=column_names)
    sample = data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    return sample

# Step 2: Outlier Detection using IQR
def remove_outliers_iqr(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[feature] = np.where(df[feature] < lower, lower,
                        np.where(df[feature] > upper, upper, df[feature]))
    return df

# Step 3: Scaling using StandardScaler
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# Full Preprocessing Pipeline
def preprocess_pipeline(filepath):
    df = load_sample_data(filepath)
    features = [col for col in df.columns if col != 'label']

    df = remove_outliers_iqr(df, features)
    df = scale_features(df, features)

    return df

# Example usage
if __name__ == "__main__":
    data_path = "HIGGS.csv"  # Replace with your local path
    processed_df = preprocess_pipeline(data_path)
    print(processed_df.head())
