# uskudar-final

# HIGGS Dataset - Machine Learning Pipeline

The project applies a full machine learning pipeline to the **HIGGS dataset**, focusing on feature selection, model training, and performance evaluation using **nested cross-validation**.

---

##  Project Objectives

- Analyze and handle outliers  
- Normalize the dataset  
- Apply **filter-based feature selection** (ANOVA, Mutual Information)  
- Train and tune models using **nested cross-validation**  
- Compare performance of multiple classifiers  
- Evaluate results using metrics and ROC curves

---

##  Dataset

- Source: [UCI ML Repository – HIGGS Dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS)  
- Size: 11 million samples, 28 features  
- Sample used: **100,000 rows** for computational manageability

---

##  Pipeline Structure

### 🔹 Section 1: Data Preprocessing
- Outlier detection using IQR
- Min-Max scaling for normalization

### 🔹 Section 2: Feature Selection
- ANOVA F-score
- Mutual Information
- Top 15 features selected by each method

### 🔹 Section 3: Nested Cross-Validation & Modeling
- **Outer loop**: 5-fold CV (for model evaluation)
- **Inner loop**: 3-fold CV (for model selection & hyperparameter tuning)
- Models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - XGBoost

---

##  Hyperparameters Tested

- **KNN**: `n_neighbors = 3 to 11`  
- **SVM**: `C = [0.1, 1, 10]`, `kernel = ['linear', 'rbf']`  
- **MLP**: `hidden_layer_sizes = [(50,), (100,)]`, `activation = ['relu', 'tanh']`  
- **XGBoost**: `n_estimators = 100`  

---

##  Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC (One-vs-All strategy)  
- ROC Curves plotted per fold + combined plot




