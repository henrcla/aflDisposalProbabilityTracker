import pandas as pd
import numpy as np
import joblib
import sys
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


def get_player_current_form(player_name):
    """
    Gets a player's current form from saved data.
    Returns rolling features based on their most recent games.
    """
    print(f"Fetching current form for {player_name}...")
    
    games_df = pd.read_csv("data/raw/all_games.csv")
    opponent_stats = pd.read_csv("data/raw/opponent_stats.csv", index_col=0)
    
    player_games = games_df[games_df["player"] == player_name].copy()
    
    if player_games.empty:
        print(f"  No data found for {player_name}")
        return None, None, None
    
    player_games = player_games[player_games["game_pct"] >= 50]
    
    def parse_round(r):
        r = str(r).strip()
        if r.isdigit(): return int(r)
        elif r == "QF": return 25
        elif r == "EF": return 25
        elif r == "SF": return 26
        elif r == "PF": return 27
        elif r == "GF": return 28
        else: return 0
    
    player_games["round_num"] = player_games["round"].apply(parse_round)
    player_games = player_games.sort_values(["season", "round_num"])
    
    last_10 = player_games.tail(10)
    last_5 = player_games.tail(5)
    
    form = {
        "rolling_avg_5": last_5["disposals"].mean(),
        "rolling_avg_10": last_10["disposals"].mean(),
        "rolling_std_5": last_5["disposals"].std(),
    }
    
    print(f"  Last 5 games: {last_5['disposals'].tolist()}")
    print(f"  Rolling avg (5): {form['rolling_avg_5']:.1f}")
    print(f"  Rolling avg (10): {form['rolling_avg_10']:.1f}")
    
    if player_name in opponent_stats.index:
        player_opponent_stats = opponent_stats.loc[player_name].to_dict()
    else:
        player_opponent_stats = {}
    
    return form, player_opponent_stats, None


def get_opponent_feature(player_name, opponent, opponent_stats):
    """
    Gets a player's career disposal average vs a specific opponent.
    Returns the career DA or None if not available.
    """
    if opponent_stats is None:
        return np.nan
    
    if opponent in opponent_stats:
        return float(opponent_stats[opponent])
    
    # Try partial match in case of naming differences
    for key in opponent_stats:
        if opponent.lower() in key.lower() or key.lower() in opponent.lower():
            return float(opponent_stats[key])
    
    return np.nan


def load_model(target):
    """Loads saved model for a given target."""
    model_path = f"models/xgb_{target}.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run model.py first.")
        return None
    return joblib.load(model_path)


def predict(player_name, opponent, round_num, game_pct=100):
    """
    Generates disposal probability predictions and implied odds
    for a player in an upcoming game.
    
    Args:
        player_name: Full player name e.g. "Clayton Oliver"
        opponent:    Opposing team e.g. "Collingwood"
        round_num:   Round number e.g. 15
        game_pct:    Expected game time percentage (default 100)
    
    Returns:
        Dictionary of predictions per disposal target
    """
    print(f"\n{'='*55}")
    print(f"  PREDICTION: {player_name}")
    print(f"  vs {opponent} | Round {round_num}")
    print(f"{'='*55}")
    
    # Get current form
    form, opponent_stats, venue_stats = get_player_current_form(player_name)
    
    if form is None:
        print("Could not generate prediction - player data unavailable.")
        return None
    
    # Get opponent feature
    career_da_vs_opponent = get_opponent_feature(
        player_name, opponent, opponent_stats
    )
    
    if np.isnan(career_da_vs_opponent):
        print(f"  No career data vs {opponent} - using rolling average")
        career_da_vs_opponent = form["rolling_avg_10"]
    else:
        print(f"  Career DA vs {opponent}: {career_da_vs_opponent:.1f}")
    
    # Build feature vector
    features = {
        "rolling_avg_5": form["rolling_avg_5"],
        "rolling_avg_10": form["rolling_avg_10"],
        "rolling_std_5": form["rolling_std_5"] if not np.isnan(
            form["rolling_std_5"]) else 0,
        "career_da_vs_opponent": career_da_vs_opponent,
        "round_num": round_num,
        "game_pct": game_pct
    }
    
    X = pd.DataFrame([features])[FEATURE_COLS]
    
    # Generate predictions for each target
    print(f"\n{'─'*55}")
    print(f"  {'TARGET':<15} {'PROBABILITY':>12} {'FAIR ODDS':>12}")
    print(f"{'─'*55}")
    
    predictions = {}
    
    for target in DISPOSAL_TARGETS:
        model = load_model(target)
        if model is None:
            continue
        
        prob = model.predict_proba(X)[0][1]
        fair_odds = 1 / prob
        
        predictions[target] = {
            "probability": prob,
            "fair_odds": fair_odds
        }
        
        print(f"  {str(target)+'+ disposals':<15} "
              f"{prob*100:>11.1f}% "
              f"{fair_odds:>11.2f}x")
    
    print(f"{'─'*55}")
    print(f"\n  HOW TO USE:")
    print(f"  If market odds > fair odds, the bet has positive expected value.")
    print(f"{'='*55}\n")
    
    return predictions


def compare_odds(predictions, market_odds):
    """
    Compares model predictions to market odds and identifies value bets.
    
    Args:
        predictions: Output from predict()
        market_odds: Dictionary like {15: 1.85, 20: 2.10, 25: 3.50}
    """
    print(f"\n{'='*55}")
    print(f"  VALUE BET ANALYSIS")
    print(f"{'='*55}")
    print(f"  {'TARGET':<15} {'FAIR ODDS':>10} {'MARKET':>10} "
          f"{'EV':>8} {'VALUE?':>8}")
    print(f"{'─'*55}")
    
    for target, pred in predictions.items():
        if target not in market_odds:
            continue
        
        fair = pred["fair_odds"]
        market = market_odds[target]
        prob = pred["probability"]
        
        # Expected value = (probability * odds) - 1
        ev = (prob * market) - 1
        has_value = market > fair
        value_str = "✓ YES" if has_value else "✗ NO"
        
        print(f"  {str(target)+'+ disposals':<15} "
              f"{fair:>10.2f} "
              f"{market:>10.2f} "
              f"{ev*100:>7.1f}% "
              f"{value_str:>8}")
    
    print(f"{'='*55}\n")


if __name__ == "__main__":
    predictions = predict(
        player_name="Clayton Oliver",
        opponent="Collingwood",
        round_num=3
    )
    
    if predictions:
        market_odds = {
            15: 1.85,
            20: 2.10,
            25: 3.50
        }
        compare_odds(predictions, market_odds)