# Hybrid Stock Forecasting: Merging Market Signals with News Sentiment

**Team:**
Likhita Nallapati | Shriya Rawal

 
---
 
## Overview
 
This project explores how news sentiment affects Apple Inc.'s stock performance by combining historical stock data with financial news headlines. The goal is to build a predictive model that forecasts future closing prices using both technical indicators and NLP-derived sentiment scores from FinBERT.
 
---
 
## Dataset Sources
 
- **Stock Prices**: Collected via `yfinance` for AAPL, spanning 2016–2024
- **News Headlines**: Financial news related to Apple from a Kaggle dataset covering the same period
 
---
 
## Methodology
 
### Exploratory Data Analysis
- Summary statistics for OHLC prices and volume
- Daily price change distribution to understand volatility
- Time series plots of closing price and trading volume
- Word clouds and frequency charts on news article titles and content
- Co-occurring stock ticker analysis (AMZN, GOOGL, international AAPL listings)
 
### Feature Engineering
- **Technical indicators**: SMA, EMA, RSI, MACD
- **Sentiment scores**: FinBERT applied to financial headlines (positive / neutral / negative)
- **Lag features**: Applied to both technical and sentiment data
- **Temporal alignment**: Sentiment and price data merged by date
 
---
 
## Modeling Phases & Results
 
### Phase 1 — Linear Regression (Baseline)
 
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Historical features only | 0.83 | 1.15 | 0.9975 |
| + Sentiment features | 0.80 | 1.11 | 0.9977 |
 
> Sentiment scores provided a modest but consistent improvement in accuracy.
 
### Phase 2 — Tree-Based Models
 
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (baseline) | 12.92 | 20.88 | 0.179 |
| Random Forest (+ lag features) | 11.19 | 19.22 | 0.304 |
| XGBoost (baseline) | 12.58 | 20.58 | 0.202 |
| XGBoost (+ lag features) | 11.62 | 19.82 | 0.260 |
 
> Tree-based models struggled with the sequential nature of time series data. Additional lag features improved performance but did not close the gap.
 
### Phase 3 — LSTM (Deep Learning)
 
- **Architecture**: 2-layer LSTM with dropout + dense output layer
- **Sequence length**: 60 days
- **Features**: All technical indicators + sentiment score
 
| R² | MAE | RMSE |
|---|---|---|
| 0.9841 | 1.0395 | 1.8510 |
 
> LSTM captured temporal dependencies most effectively, outperforming all other models.
 
---
 
## Key Findings
 
- **Sentiment features help**: Modest but consistent improvement across all model types
- **Linear models surprisingly strong**: Due to the stability and trend consistency of AAPL prices
- **LSTM best overall**: Deep learning was the most suitable approach for sequential financial forecasting
 
---
 
## Repo Structure
 
| File | Description |
|---|---|
| `ReadMe.md` | Project summary, methodology, and results walkthrough |
| `EDA` | Exploratory data analysis — trends, correlations, visualizations |
| `Preprocessing and Feature Engineering` | Data cleaning, transformation, and technical feature engineering |
| `models.ipynb` | Model implementation, training, evaluation, and visualization |
 
**Datasets** (not included in repo due to size — see sources below):
- `historical_data.csv` — Raw AAPL stock data from Yahoo Finance
- `apple_news_data.csv` — Raw financial news headlines
- `news_with_sentiment.csv` — Headlines with FinBERT sentiment scores
- `merged_data.csv` — Final combined dataset ready for modeling
 
---
 
## Limitations
 
- Relatively small dataset (~27,000 rows)
- Tree-based models required additional feature engineering to handle time-series structure
- Analysis limited to a single stock (AAPL) over a defined time window
- Reliance on historical data limits performance during unprecedented market events
 
---
 
## Tools & Libraries
 
`Python` · `pandas` · `yfinance` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `FinBERT` · `matplotlib` · `seaborn`
