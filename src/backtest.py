import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

DISPOSAL_TARGETS = [15, 20, 25]
FEATURE_COLS = [
    "rolling_avg_5",
    "rolling_avg_10",
    "rolling_std_5",
    "career_da_vs_opponent",
    "round_num",
    "game_pct"
]


def load_model(target):
    """Loads the saved model for a given target."""
    model = joblib.load(f"models/xgb_{target}.joblib")
    return model


def run_backtest(target):
    """
    Proper time-based backtest.
    Trains on 2022-2023, tests on 2024 data only.
    """
    print(f"\n{'='*50}")
    print(f"BACKTEST: {target}+ disposals")
    print(f"{'='*50}")
    
    df = pd.read_csv(f"data/raw/features_{target}.csv")
    
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    df = df.dropna(subset=["rolling_avg_5", "rolling_std_5",
                            "round_num", "game_pct"])
    
    # Time based split - train on 2022-2023, test on 2024
    train_df = df[df["season"] <= 2023]
    test_df = df[df["season"] == 2024]
    
    print(f"Training games: {len(train_df)}")
    print(f"Test games (2024): {len(test_df)}")
    
    # Train fresh model on training data only
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[f"hit_{target}_disposals"]
    model.fit(X_train, y_train)
    
    # Predict on 2024 test data only
    X_test = test_df[FEATURE_COLS]
    test_df = test_df.copy()
    test_df["predicted_prob"] = model.predict_proba(X_test)[:, 1]
    test_df["actual"] = test_df[f"hit_{target}_disposals"]
    test_df["fair_odds"] = 1 / test_df["predicted_prob"]
    
    # Calibration on test set
    print("\nCalibration check (2024 test data):")
    test_df["prob_bucket"] = pd.cut(test_df["predicted_prob"],
                                     bins=[0, 0.2, 0.3, 0.4, 0.5,
                                           0.6, 0.7, 0.8, 1.0])
    calibration = test_df.groupby("prob_bucket", observed=True).agg(
        predicted=("predicted_prob", "mean"),
        actual=("actual", "mean"),
        count=("actual", "count")
    ).round(3)
    print(calibration)
    
    # Value betting simulation on test data
    # Only bet when model probability differs significantly from 50/50
    # i.e. when the model has conviction
    print("\nValue betting simulation (2024 only):")
    print("(Only betting when model probability > 60% or < 40%)")
    
    # High conviction bets only
    conviction_bets = test_df[
        (test_df["predicted_prob"] > 0.60) | 
        (test_df["predicted_prob"] < 0.40)
    ].copy()
    
    # For high probability predictions, bet YES
    yes_bets = conviction_bets[conviction_bets["predicted_prob"] > 0.60].copy()
    yes_bets["market_odds"] = yes_bets["fair_odds"] * 1.10
    yes_bets["profit"] = np.where(
        yes_bets["actual"] == 1,
        yes_bets["market_odds"] - 1,
        -1
    )
    
    # For low probability predictions, bet NO (fade the player)
    no_bets = conviction_bets[conviction_bets["predicted_prob"] < 0.40].copy()
    no_bets["market_odds"] = (1 / (1 - no_bets["predicted_prob"])) * 1.10
    no_bets["profit"] = np.where(
        no_bets["actual"] == 0,
        no_bets["market_odds"] - 1,
        -1
    )
    
    all_bets = pd.concat([yes_bets, no_bets])
    
    print(f"\n  High conviction bets: {len(all_bets)}")
    print(f"  Yes bets: {len(yes_bets)}, No bets: {len(no_bets)}")
    
    if len(all_bets) > 0:
        total_profit = all_bets["profit"].sum()
        roi = (total_profit / len(all_bets)) * 100
        print(f"  Total profit: ${total_profit:.2f}")
        print(f"  ROI: {roi:.1f}%")
    
    return test_df


def plot_calibration(df, target):
    """
    Plots predicted probability vs actual hit rate.
    A well calibrated model should follow the diagonal line.
    """
    os.makedirs("results/figures", exist_ok=True)
    
    buckets = df.groupby("prob_bucket", observed=True).agg(
        predicted=("predicted_prob", "mean"),
        actual=("actual", "mean")
    ).dropna()
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(buckets["predicted"], buckets["actual"],
             "bo-", label="Model")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Hit Rate")
    plt.title(f"Calibration Plot - {target}+ Disposals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/calibration_{target}.png")
    plt.close()
    print(f"\nCalibration plot saved to results/figures/calibration_{target}.png")


if __name__ == "__main__":
    for target in DISPOSAL_TARGETS:
        df = run_backtest(target)
        plot_calibration(df, target)