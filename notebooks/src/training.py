import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from src.metrics import average_f1_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from datetime import time

SEED=42


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                             model_type: str, model_params: dict,
                             cv_splits: int, random_seed: int = SEED):
    """
    Trains a specified model using StratifiedGroupKFold and evaluates using the custom F1 metric.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Encoded target variable.
        groups (pd.Series): Grouping variable for StratifiedGroupKFold (e.g., 'subject').
        model_type (str): Type of model ('catboost', 'xgboost', 'lightgbm').
        model_params (dict): Parameters for the model.
        cv_splits (int): Number of CV splits.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing model name, mean F1 score, std F1 score, and list of individual scores.
    """
    print(f"  Starting training and evaluation for {model_type.upper()}...")
    
    # State management : Validate Inputs
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("Length of X, y and groups must be equal.")
    # Assuming MODEL_PARAMS are passed globally in notebooks.
    if model_type not in MODEL_PARAMS: 
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_PARAMS.keys())}")
    
    skf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)
    
    f1_scores = []
    models = [] # store for FI and PI
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        print(f"    Fold {fold + 1}/{cv_splits}...")
        start_time = time.time()
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- Initialize Model ---
        if model_type == 'catboost':
            model = cat.CatBoostClassifier(**model_params)
            # CatBoost can handle categorical features if specified, but for now assuming all numeric
            model.fit(X_train_fold, y_train_fold, 
                      eval_set=(X_val_fold, y_val_fold), 
                      early_stopping_rounds=50, # Add early stopping
                      verbose=False)
        elif model_type == 'light_gbm':
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        else:
            raise ValueError(f"Model type {model_type} not handled in training loop.")
        
        models.append(model) # Store the trained model
        
        # --- Predict Probabilities ---
        y_pred_proba = model.predict_proba(X_val_fold)
        
        # --- Evaluate using Custom Metric ---
        try:
            fold_f1 = average_f1_score(y_val_fold, y_pred_proba)
            f1_scores.append(fold_f1)
            print(f"      Fold {fold + 1} F1 Score: {fold_f1:.4f}")
        except Exception as e:
            print(f"      Error calculating F1 for fold {fold + 1}: {e}")
            f1_scores.append(np.nan) # Append NaN if calculation fails
            
        end_time = time.time()
        fold_times.append(end_time - start_time)
    
    # --- Calculate Final Metrics ---
    mean_f1 = np.nanmean(f1_scores) # Use nanmean in case of any NaNs
    std_f1 = np.nanstd(f1_scores)
    
    print(f"  Completed {model_type.upper()} training. Mean F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
    print(f"  Average Fold Time: {np.mean(fold_times):.2f}s")
    
    return {
        'model_name': model_type.upper(),
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'fold_scores': f1_scores,
        'models': models # Return models for potential later use
    }



