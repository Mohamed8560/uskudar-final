# feature_selection.py

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# ANOVA F-Score Feature Selection
def select_features_anova(X, y, k=15):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

# Mutual Information Feature Selection
def select_features_mi(X, y, k=15):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

# Example usage
if __name__ == "__main__":
    from higgs_preprocessing import preprocess_pipeline

    # Load preprocessed data
    df = preprocess_pipeline("HIGGS.csv")  # Replace with your actual path
    X = df.drop("label", axis=1)
    y = df["label"]

    # Select features using ANOVA
    X_anova, anova_features = select_features_anova(X, y)
    print("Top 15 features (ANOVA):", list(anova_features))

    # Select features using Mutual Information
    X_mi, mi_features = select_features_mi(X, y)
    print("Top 15 features (Mutual Information):", list(mi_features))
