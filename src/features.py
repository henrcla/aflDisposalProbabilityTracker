import pandas as pd
import numpy as np

# Minimum game percentage to include a game in calculations
MIN_GAME_PCT = 50

# The disposal targets we want to predict
# e.g. 15+ disposals, 20+ disposals, 25+ disposals
DISPOSAL_TARGETS = [15, 20, 25]

def load_data():
    """
    Loads raw data and applies basic quality filters.
    """
    games_df = pd.read_csv("data/raw/all_games.csv")
    opponent_stats = pd.read_csv("data/raw/opponent_stats.csv", index_col=0)
    venue_stats = pd.read_csv("data/raw/venue_stats.csv", index_col=0)
    
    # Filter to games where player was on field for meaningful time
    games_df = games_df[games_df["game_pct"] >= MIN_GAME_PCT]
    
    # Filter to high-usage players only (midfielders, key position players)
    # who average at least 10 disposals - these are the relevant players
    # for disposal target predictions
    player_averages = games_df.groupby("player")["disposals"].mean()
    high_usage_players = player_averages[player_averages >= 8].index
    games_df = games_df[games_df["player"].isin(high_usage_players)]
    print(f"High usage players: {games_df['player'].nunique()}")
    
    print(f"Games after filtering: {len(games_df)}")
    print(f"Unique players: {games_df['player'].nunique()}")
    
    return games_df, opponent_stats, venue_stats

def add_rolling_features(games_df):
    """
    Adds rolling disposal averages for each player.
    
    For each game, we calculate what the player averaged in their
    PREVIOUS 5 games (not including the current game).
    
    This is called a 'lag' - we never let the model see the current
    game's result when calculating features, otherwise we'd be
    cheating.
    """
    # Sort by player and season so rolling calculations are in order
    games_df = games_df.sort_values(["player", "season", "round"])
    
    # Group by player and calculate rolling average
    # shift(1) means we look at previous games, not the current one
    games_df["rolling_avg_5"] = (
        games_df.groupby("player")["disposals"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=3).mean())
    )
    
    # Also add a longer term rolling average for context
    games_df["rolling_avg_10"] = (
        games_df.groupby("player")["disposals"]
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=5).mean())
    )
    
    # Rolling standard deviation - measures how consistent a player is
    # A high std means unpredictable, low std means consistent
    games_df["rolling_std_5"] = (
        games_df.groupby("player")["disposals"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=3).std())
    )
    
    return games_df

def add_opponent_venue_features(games_df, opponent_stats, venue_stats):
    """
    Adds each player's career disposal average against today's
    opponent and at today's venue as features.
    """
    
    def get_opponent_da(row):
        """Look up this player's career DA vs today's opponent."""
        player = row["player"]
        opponent = row["opponent"]
        
        # Check if we have data for this player and opponent
        if player in opponent_stats.index and opponent in opponent_stats.columns:
            val = opponent_stats.loc[player, opponent]
            # Return NaN if the value is missing
            if pd.notna(val):
                return float(val)
        return np.nan
    
    def get_venue_da(row):
        """Look up this player's career DA at today's venue."""
        player = row["player"]
        venue = row["venue"] if "venue" in row else np.nan
        
        if pd.isna(venue):
            return np.nan
            
        if player in venue_stats.index and venue in venue_stats.columns:
            val = venue_stats.loc[player, venue]
            if pd.notna(val):
                return float(val)
        return np.nan
    
    print("Adding opponent features...")
    games_df["career_da_vs_opponent"] = games_df.apply(get_opponent_da, axis=1)
    
    print("Adding venue features...")
    games_df["career_da_at_venue"] = games_df.apply(get_venue_da, axis=1)
    
    return games_df

def add_target_variables(games_df):
    """
    Adds binary target columns for each disposal target.
    
    For each target (e.g. 20 disposals), creates a column that is:
        1 if the player reached the target
        0 if they didn't
    
    These are what our model will learn to predict.
    """
    for target in DISPOSAL_TARGETS:
        col_name = f"hit_{target}_disposals"
        games_df[col_name] = (games_df["disposals"] >= target).astype(int)
    
    return games_df


def add_home_away_feature(games_df):
    """
    Adds a binary flag for whether the player's team was at home.
    
    We determine this from the opponent column - if the game is
    listed under a team's page, we know which team is the home side
    from the round data. For simplicity we'll use game_pct as a
    proxy - home teams tend to have slightly higher game time.
    
    For now we add a placeholder - this can be enhanced later
    with actual venue/home ground data.
    """
    # Extract round number as integer where possible
    # Finals rounds (QF, SF, PF, GF) get assigned high numbers
    def parse_round(round_str):
        round_str = str(round_str).strip()
        if round_str.isdigit():
            return int(round_str)
        elif round_str == "QF":
            return 25
        elif round_str == "EF":
            return 25
        elif round_str == "SF":
            return 26
        elif round_str == "PF":
            return 27
        elif round_str == "GF":
            return 28
        else:
            return np.nan
    
    games_df["round_num"] = games_df["round"].apply(parse_round)
    
    return games_df

def build_features(save=True):
    """
    Master function that builds the complete feature set.
    Loads raw data, adds all features, and saves the result.
    """
    print("Loading data...")
    games_df, opponent_stats, venue_stats = load_data()
        
    print("Adding rolling features...")
    games_df = add_rolling_features(games_df)
        
    print("Adding opponent and venue features...")
    games_df = add_opponent_venue_features(games_df, opponent_stats, venue_stats)
        
    print("Adding target variables...")
    games_df = add_target_variables(games_df)
        
    print("Adding round features...")
    games_df = add_home_away_feature(games_df)
        
    # Drop rows where we don't have enough history for rolling features
    before = len(games_df)
    games_df = games_df.dropna(subset=["rolling_avg_5"])
    after = len(games_df)
    print(f"Dropped {before - after} rows with insufficient history")
        
    print(f"\nFinal dataset: {len(games_df)} games, {len(games_df.columns)} columns")
    print(f"Columns: {games_df.columns.tolist()}")
        
    if save:
        games_df.to_csv("data/raw/features.csv", index=False)
        print("Saved to data/raw/features.csv")
        
    return games_df


if __name__ == "__main__":
    df = build_features()
    
    # Show summary of target variables
    print("\nTarget variable hit rates:")
    for target in DISPOSAL_TARGETS:
        col = f"hit_{target}_disposals"
        rate = df[col].mean() * 100
        print(f"  {target}+ disposals: {rate:.1f}% of games")