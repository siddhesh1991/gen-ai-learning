import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier, plot_importance

# Function to generate synthetic dataset
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=25, n_informative=10, n_redundant=5, random_state=42)
    return pd.DataFrame(X), pd.Series(y)

# Function for initial filtering (Variance Threshold and Correlation)
def initial_filtering(X):
    # Variance Threshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_var = sel.fit_transform(X)
    X_var = pd.DataFrame(X_var, columns=X.columns[sel.get_support(indices=True)])

    # Correlation Matrix
    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    
    # Drop features based on correlation
    X_filtered = X_var.drop(columns=to_drop)
    
    # Visualizing Correlation Matrix after dropping
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_filtered.corr().abs(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix after Filtering')
    plt.show()
    
    return X_filtered

# Function for univariate feature selection
def univariate_selection(X, y):
    sel = SelectKBest(f_classif, k=10)
    X_selected = sel.fit_transform(X, y)
    X_selected = pd.DataFrame(X_selected, columns=X.columns[sel.get_support(indices=True)])
    return X_selected, sel

# Function for Recursive Feature Elimination with Cross-Validation
def recursive_feature_elimination(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='recall')
    rfecv.fit(X, y)
    
    plt.figure()
    plt.title('RFECV - Number of features vs Recall score')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (recall)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return rfecv

# Main function to run the feature selection process
def run_feature_selection():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_filtered = initial_filtering(X_train)
    X_uni, sel_univariate = univariate_selection(X_filtered, y_train)
    rfecv = recursive_feature_elimination(X_uni, y_train)
    
    # Apply selection to training and testing sets
    X_train_selected = rfecv.transform(X_uni)
    X_test_filtered = initial_filtering(X_test)
    X_test_uni = sel_univariate.transform(X_test_filtered)
    X_test_selected = rfecv.transform(X_test_uni)

    # Train and evaluate the final model
    final_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    final_model.fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)
    final_recall = recall_score(y_test, y_pred)
    print(f"Final Recall Score: {final_recall:.4f}")

# Execute the function
run_feature_selection()
