# Kaggle Competition Skills — Condensed Summary

## Quick Decision Tree
```
Is it tabular data?
  YES → AutoGluon first → CatBoost → Blend
  NO → Is it time series?
    YES → TimeSeriesPredictor + Chronos-2
    NO → Is it a code competition?
      YES → Fork public baseline + artifacts
      NO → Is it simulation (TrueSkill)?
        YES → CPU eval harness + latest-N strategy
```

## Top 5 Rules
1. AutoGluon baseline FIRST (5 min > 5 hours manual)
2. ONE variable per experiment
3. OOF ≠ LB (check gap, especially small data)
4. 3× <0.0001 → pivot
5. Extract skill after every experiment

## Top 5 Dead Ends (Avoid!)
1. analyst D=0 on all regions (platform rejects)
2. Foundation model on retail sales (Chronos-2 < LightGBM+lag for structured lags)
3. Ensemble with correlation > 0.97 (signal dilution)
4. Recursive forecasting for >7 steps (error accumulation)
5. Feature engineering without LB validation (OOF overfit)
