# OOF vs LB Gap — Critical Awareness

## The Problem
Out-of-fold (OOF) cross-validation scores do NOT always match leaderboard (LB) scores.

## When Gap is Large
- Small data (N < 1,000): OOF can overestimate by 0.03-0.09
- Time series multi-horizon: OOF uses actual lags, LB needs recursive prediction
- Severe class imbalance: stratified CV may not reflect test distribution

## Decision Rules
1. OOF↑ LB↑ → Real improvement
2. OOF↑ LB↓ → Overfitting (STOP adding features)
3. OOF↓ LB↑ → CV noise (feature may still be useful)
4. OOF↓ LB↓ → Feature is harmful

## Validation
- Always use TimeSeriesSplit for time series (never KFold)
- For multi-horizon forecasting: validation lag features ≠ test lag features
- Per-family × per-day error decomposition reveals hidden issues
