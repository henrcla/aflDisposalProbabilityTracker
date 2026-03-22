# AFL Disposal Probability Predictor

A machine learning pipeline that predicts the probability of AFL players 
reaching specific disposal targets (15+, 20+, 25+) in upcoming matches, 
and identifies value bets by comparing model probabilities to market odds.

## Overview

This project scrapes historical AFL player data, engineers predictive 
features, trains calibrated classification models, and generates 
match-specific probability estimates with implied fair odds.

Built as a demonstration of applied ML in a sports analytics context — 
the core methodology (probability estimation, expected value calculation, 
backtesting) is directly analogous to quantitative strategies used in 
financial markets.

## Results

| Target | Model | AUC | Brier Score |
|--------|-------|-----|-------------|
| 15+ disposals | XGBoost | 0.783 | 0.190 |
| 20+ disposals | XGBoost | 0.806 | 0.179 |
| 25+ disposals | XGBoost | 0.812 | 0.164 |

**Backtesting on unseen 2024 data:** 4.5–5.8% ROI on high-conviction 
predictions, with well-calibrated probability estimates across all targets.

## Example Output
```
=======================================================
  PREDICTION: Clayton Oliver
  vs Collingwood | Round 3
=======================================================
  Last 5 games: [27, 28, 30, 26, 26]
  Rolling avg (5): 27.4
  Career DA vs Collingwood: 28.5
───────────────────────────────────────────────────────
  TARGET           PROBABILITY    FAIR ODDS
───────────────────────────────────────────────────────
  15+ disposals          95.1%        1.05x
  20+ disposals          96.0%        1.04x
  25+ disposals          80.2%        1.25x
───────────────────────────────────────────────────────
```

## Features Used

| Feature | Description |
|---------|-------------|
| `rolling_avg_5` | Player's disposal average over last 5 games |
| `rolling_avg_10` | Player's disposal average over last 10 games |
| `rolling_std_5` | Disposal consistency over last 5 games |
| `career_da_vs_opponent` | Career disposal average vs today's opponent |
| `round_num` | Round number (season progression effects) |
| `game_pct` | Percentage of game played |

## Pipeline Architecture
```
Data Collection     Feature Engineering     Modelling          Prediction
─────────────      ───────────────────     ─────────          ──────────
afltables.com  →   Rolling averages     →  XGBoost        →   Probability
(88,741 games,     Opponent history        Logistic Reg.      Fair odds
 950 players)      Consistency metrics     Cross-validation   EV analysis
                   Target encoding         Backtesting
```

## Project Structure
```
afl-disposal-predictor/
├── src/
│   ├── data_loader.py   # Web scraping pipeline (requests, BeautifulSoup)
│   ├── features.py      # Feature engineering (pandas, numpy)
│   ├── model.py         # Model training and evaluation (XGBoost, sklearn)
│   ├── backtest.py      # Time-based backtesting framework
│   └── predict.py       # Live prediction and odds comparison
├── data/
│   └── raw/             # Scraped and processed data (gitignored)
├── results/
│   └── figures/         # Calibration plots
└── models/              # Saved models (gitignored)
```

## Methodology

### Data Collection
Player game-by-game statistics scraped from afltables.com covering 
88,741 games across 950 players. Each player's career history, 
opponent-specific averages, and venue averages are extracted.

### Feature Engineering
Rolling features use a **lag of 1** to prevent data leakage — when 
predicting a game, only information available before that game is used. 
Player pools are filtered per target to include only players for whom 
that target is statistically meaningful.

### Modelling
Three separate XGBoost classifiers trained, one per disposal target. 
Evaluated with 5-fold stratified cross-validation. Logistic regression 
used as baseline — AUC scores within 0.003 of XGBoost suggest the 
relationship is largely linear, meaning features are the performance 
bottleneck rather than model complexity.

### Backtesting
Time-based train/test split: trained on data up to 2023, tested on 
2024 season only. High-conviction predictions (model probability >60% 
or <40%) achieve 4.5–5.8% ROI, suggesting genuine predictive signal 
beyond the baseline hit rate.

### Probability Calibration
Calibration plots confirm predicted probabilities closely match actual 
hit rates across all probability buckets — essential for fair odds 
calculation to be meaningful.

## Usage

### Setup
```bash
git clone https://github.com/henrcla/aflDisposalProbabilityTracker
cd aflDisposalProbabilityTracker
pip install -r requirements.txt
```

### Scrape data (first time, ~40 mins)
```bash
python src/data_loader.py
```

### Update current season data (weekly, ~5 mins)
```bash
python src/data_loader.py update
```

### Build features and train models
```bash
python src/features.py
python src/model.py
```

### Generate prediction
Edit the bottom of `src/predict.py` with the player, opponent and round,
then run:
```bash
python src/predict.py
```

### Run backtest
```bash
python src/backtest.py
```

## Tech Stack

- **Python 3.12**
- **Data:** requests, BeautifulSoup4, pandas, numpy
- **Modelling:** scikit-learn, XGBoost
- **Visualisation:** matplotlib, seaborn

## Future Improvements

- Add venue data to game-by-game records for venue feature
- Platt scaling for improved probability calibration in low-probability buckets
- Expand to kicks, handballs, marks and other player prop targets
- REST API wrapper for programmatic access to predictions
- Automated weekly data refresh and model retraining