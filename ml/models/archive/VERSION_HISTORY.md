# Model & Policy Version History

## Active Canonical Baseline: v3.1
- **Model Version:** `v3.1` (HPO-optimized LightGBM, Trial #47)
- **Calibration Version:** `oof-iso-v3.1` (5-Fold Stratified OOF Isotonic Regression)
- **Policy Version:** `v3.1-policy-v1` (Approve <= 0.045, Reject >= 0.335, Review Rate: 24.09%)
- **Test Metrics (Untouched Test Set N=21,398):**
  - ROC-AUC: 0.8615
  - PR-AUC: 0.3995
  - Brier Score: 0.0492
  - Weighted ECE: 0.0012
  - Macro ECE: 0.0129
  - Expected Economic Cost: $4,062,100 ($189.84 / applicant)
- **Bootstrap Uncertainty (95% CI vs v3.0.0):**
  - Δ ROC-AUC: [+0.0002, +0.0032] (Statistically distinguishable)
  - Δ Brier: [-0.0005, -0.0001] (Statistically distinguishable)
  - Δ PR-AUC: [-0.0011, +0.0104] (Includes zero)
  - Δ Total Cost: [-$153,881, +$109,024] (Includes zero — cost improvement not statistically distinguishable)

---

## Historical Archived Baselines

### v3.0.0 (Archived in `ml/models/archive/v3.0.0/`)
- **Model Version:** `v3.0.0-oof-baseline`
- **Calibration Version:** `oof-iso-v3.0`
- **Policy Version:** `frozen_oof_3tier_policy` (Approve <= 0.055, Reject >= 0.405, Review Rate: 24.61%)
- **Hyperparameters:** `n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31`
- **Test Metrics:**
  - ROC-AUC: 0.8599 | PR-AUC: 0.3947 | Brier: 0.0494 | wECE: 0.0018 | Expected Cost: $4,082,900
