# inner_cv.py

# nested_cv_modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore")






from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline

def inner_cv_function(X, y, k_features=15, inner_splits=3):
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    feature_methods = {
        "anova": f_classif,
        "mutual_info": mutual_info_classif
    }

    models = {
        "knn": (KNeighborsClassifier, {"n_neighbors": list(range(3, 12, 2))}),
        "svm": (SVC, {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "probability": [True]}),
        "mlp": (MLPClassifier, {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"], "max_iter": [500]}),
        "xgb": (XGBClassifier, {"n_estimators": [100], "use_label_encoder": [False], "eval_metric": ["logloss"]})
    }

    best_score = 0
    best_model = None
    best_features = None

    for feat_name, feat_func in feature_methods.items():
        selector = SelectKBest(score_func=feat_func, k=k_features)
        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()]

        for model_name, (model_class, param_grid) in models.items():
            for params in ParameterGrid(param_grid):
                val_scores = []

                for train_idx, val_idx in inner_cv.split(X_selected, y):
                    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = model_class(**params)
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    val_scores.append(score)

                mean_val_score = np.mean(val_scores)

                if mean_val_score > best_score:
                    best_score = mean_val_score
                    best_model = model_class(**params).fit(X_selected, y)  # Refit on full train_val
                    best_features = selected_feature_names

    return best_model, best_features
