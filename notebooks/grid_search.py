import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
from joblib import Parallel, delayed
import numpy as np

def train_and_evaluate(params, X_train, X_val, y_train, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=10, tree_method='gpu_hist')
    
    # Create a threshold list
    thresholds = np.linspace(0, 1, 101)
    
    best_threshold = None
    best_recall = 0
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = model.predict(dval) > threshold
        
        recall = recall_score(y_val, y_pred)
        if recall >= 0.9 and recall > best_recall:
            best_recall = recall
            precision = precision_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            best_metrics = {'recall': recall, 'precision': precision, 'auc': auc, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            best_threshold = threshold
    
    return best_metrics, best_threshold

def grid_search(X_train, X_val, y_train, y_val):
    # Create the grid search matrix
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        # Add other XGBoost parameters here
    }
    params_list = list(ParameterGrid(param_grid))
    
    # Parallel execution of train_and_evaluate
    results = Parallel(n_jobs=-1, prefer='threads')(delayed(train_and_evaluate)(params, X_train, X_val, y_train, y_val) for params in params_list)
    
    # Convert results to a DataFrame
    metrics_list = [result[0] for result in results]
    threshold_list = [result[1] for result in results]
    performance_df = pd.DataFrame(metrics_list, index=pd.MultiIndex.from_tuples(params_list, names=param_grid.keys()))
    performance_df['threshold'] = threshold_list
    
    return performance_df

# Assuming you have your data loaded as X and y
# Split the data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Run the grid search
performance_df = grid_search(X_train, X_val, y_train, y_val)