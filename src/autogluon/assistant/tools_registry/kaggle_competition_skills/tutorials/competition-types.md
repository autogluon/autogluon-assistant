# Competition Type Strategy Guide

## Standard Tabular
- AutoGluon first → CatBoost second → Blend
- Key skills: autogluon-first, catboost-first, kaggle-optimal-blending

## Code Competition
- Fork public baseline (80% of work)
- Add artifact datasets as kernel inputs
- Trim non-essential pipeline cells
- Key skill: code-competition-artifact-pipeline

## Simulation Competition (TrueSkill)
- Re-submit = reset convergence (μ₀=600)
- Latest-N rule: only recent submissions count
- CPU eval harness to test before submitting
- Key skill: trueskill-simulation-competition-strategy

## Time Series
- Use TimeSeriesPredictor (NOT TabularPredictor)
- Chronos-2: known_covariates support (1.5 only)
- For retail data: LightGBM + lag features may beat foundation models
- Multi-horizon: recursive has error accumulation, LOCF is safer
- Key skill: autogluon-timeseries-strategy

## ONNX Optimization
- Convolution weights can be solved via LogisticRegression
- Sparse initializers rejected by strict ONNX checker
- Gather operations for non-trivial patterns
- Key skill: onnx-minimal-network-design
