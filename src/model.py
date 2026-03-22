# StratifiedKFold — splits data into train/test folds while preserving the ratio of positive to negative examples in each fold. 
# Important for imbalanced targets like ours.
# roc_auc_score — measures how well our model separates hits from misses. 0.5 means no better than random, 1.0 means perfect. 
# We're aiming for 0.65+.
# brier_score_loss — measures the accuracy of our probability predictions specifically. Lower is better.
# This is important because we're not just predicting yes/no — we're predicting a probability.
# LogisticRegression — our baseline model. Always build a simple baseline before a complex model 
# so you know if XGBoost is actually adding value.

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Features used to train model
FEATURE_COLS = [
    "rolling_avg_5",        # Player's recent form
    "rolling_avg_10",       # Player's longer term form
    "rolling_std_5",        # Player's consistency
    "career_da_vs_opponent", # Historical performance vs this opponent
    "round_num",            # Where we are in the season
    "game_pct"              # How much of the game they played
]

DISPOSAL_TARGETS = [15, 20, 25]

RESULTS_PATH = "results/"

def load_target_data(target):
    """
    Loads the feature dataset for a specific disposal target.
    Returns X (features) and y (target variable) ready for modelling.
    """
    path = f"data/raw/features_{target}.csv"
    df = pd.read_csv(path)
    
    # Only drop rows missing essential features
    essential_cols = ["rolling_avg_5", "rolling_std_5", 
                      "round_num", "game_pct"]
    df = df.dropna(subset=essential_cols)
    
    # Fill optional features with median where missing
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[FEATURE_COLS]
    y = df[f"hit_{target}_disposals"]
    
    print(f"\nTarget {target}+ disposals:")
    print(f"  Total samples: {len(df)}")
    print(f"  Hit rate: {y.mean()*100:.1f}%")
    print(f"  Features: {X.shape[1]}")
    
    return X, y, df

def evaluate_model(model, X, y, model_name, target):
    """
    Evaluates a model using cross validation.
    
    We use cross validation rather than a single train/test split
    because it gives a more reliable estimate of real performance
    by testing on multiple different held-out subsets.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # AUC score - how well does the model rank predictions
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    
    # Brier score - how accurate are the probabilities
    brier_scores = cross_val_score(model, X, y, cv=cv, 
                                   scoring="neg_brier_score")
    
    print(f"\n  {model_name}:")
    print(f"    AUC:         {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
    print(f"    Brier Score: {(-brier_scores.mean()):.3f} (+/- {brier_scores.std():.3f})")
    
    return {
        "model_name": model_name,
        "target": target,
        "auc_mean": auc_scores.mean(),
        "auc_std": auc_scores.std(),
        "brier_mean": -brier_scores.mean(),
        "brier_std": brier_scores.std()
    }

def train_and_evaluate(target):
    """
    Trains and evaluates both baseline and XGBoost models
    for a given disposal target.
    """
    print(f"\n{'='*50}")
    print(f"DISPOSAL TARGET: {target}+")
    print(f"{'='*50}")
    
    X, y, df = load_target_data(target)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    
    # Baseline - Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    results.append(evaluate_model(lr, X_scaled, y, 
                                  "Logistic Regression", target))
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    results.append(evaluate_model(xgb_model, X, y, 
                                  "XGBoost", target))
    
    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    all_results = []
    
    for target in DISPOSAL_TARGETS:
        results = train_and_evaluate(target)
        all_results.extend(results)
    
    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    results_df = pd.DataFrame(all_results)
    print(results_df[["target", "model_name", "auc_mean", "brier_mean"]].to_string(index=False))