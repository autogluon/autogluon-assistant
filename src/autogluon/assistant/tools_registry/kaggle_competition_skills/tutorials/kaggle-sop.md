# Kaggle Experiment Standard Operating Procedure

## Phase 0: Pre-Experiment
- Identify competition type (tabular / code / simulation / research)
- Check metric (accuracy / RMSE / AUC / custom)
- Check data size (determines OOF/LB gap expectation)
- Identify GPU requirements

## Phase 1: Baseline
1. Run AutoGluon baseline (5-15 min)
2. Submit to LB — this is your floor score
3. Check feature_importance() to identify top features

## Phase 2: Incremental Experiments
- ONE variable per experiment
- Always record: OOF score, LB score, gap
- If OOF improves but LB doesn't → overfitting CV
- 3 consecutive improvements < 0.0001 → pivot

## Phase 3: Ensemble
- Check model correlation (must be < 0.93 for ensemble benefit)
- Nelder-Mead optimize blend weights on OOF
- Multi-seed averaging for variance reduction

## Phase 4: Knowledge Crystallization
- Success → extract skill (what worked + why + when to apply)
- Failure → record dead-end (direction + failure mode + evidence)
- Update memory vault
