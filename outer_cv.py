# outer_cv.py



from roc_plotter import plot_roc_ova







from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def outer_cv_evaluation(X, y, inner_cv_function, n_splits=5, random_state=42):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = []  # To store all performance metrics for each outer fold

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n🔁 Outer Fold {fold_idx+1}")
         # After predicting probabilities:
        plot_roc_ova(y_test, y_proba.reshape(-1, 1), fold_idx + 1)

        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

        # Call the inner CV logic to get the best trained model
        best_model, selected_features = inner_cv_function(X_train_val, y_train_val)

        # Apply same feature selection on test data
        X_test_selected = X_test[selected_features]

        # Predict and evaluate
        y_pred = best_model.predict(X_test_selected)
        y_proba = best_model.predict_proba(X_test_selected)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        
       

             

        

        metrics = {
            "fold": fold_idx + 1,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }

        results.append(metrics)
        print(f"✅ Fold {fold_idx+1} Metrics: {metrics}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\n📊 Cross-Validation Summary:")
    print(results_df.mean(numeric_only=True))


    
    
    return results_df
