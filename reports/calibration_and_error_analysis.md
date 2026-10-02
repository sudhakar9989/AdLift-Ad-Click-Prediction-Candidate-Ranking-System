# AdLift Calibration, Slicing & Failure Analysis

**Evaluated Dataset Split**: `validation.csv` (15,000 rows, chronological held-out split).

- **Empirical Validation Click Rate**: 34.29%
- **Mean Model Predicted Probability**: 35.53%
- **Expected Calibration Error (ECE)**: 0.0175
- **Brier Score**: 0.2063

## 1. Score Deciles & Observed Click Rate

| Score Decile | Count | % of Val | Observed CTR | Mean Predicted | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [0.0, 0.1) | 30 | 0.2% | 10.00% | 9.29% | +0.0071 |
| [0.1, 0.2) | 1,946 | 12.97% | 15.47% | 16.35% | -0.0088 |
| [0.2, 0.3) | 4,184 | 27.89% | 25.76% | 25.22% | +0.0055 |
| [0.3, 0.4) | 3,775 | 25.17% | 33.11% | 34.76% | -0.0165 |
| [0.4, 0.5) | 2,579 | 17.19% | 41.61% | 44.61% | -0.0300 |
| [0.5, 0.6) | 1,436 | 9.57% | 49.86% | 54.54% | -0.0468 |
| [0.6, 0.7) | 728 | 4.85% | 64.70% | 64.26% | +0.0043 |
| [0.7, 0.8) | 291 | 1.94% | 77.32% | 73.81% | +0.0351 |
| [0.8, 0.9) | 31 | 0.21% | 87.10% | 82.15% | +0.0495 |

## 2. Performance Slices on Known Fields

### Recency Slices (`time_since_last_click`)

| Slice | Count | % of Val | Actual CTR | Mean Pred | ROC AUC | Log Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **First-Time User (No prior click)** | 7,559 | 50.39% | 29.41% | 30.20% | 0.6363 | 0.5821 |
| **Returning Active (< 1 hour)** | 739 | 4.93% | 66.04% | 62.38% | 0.6605 | 0.6061 |
| **Returning Dormant (> 24 hours)** | 5,442 | 36.28% | 33.94% | 35.64% | 0.6452 | 0.6102 |
| **Returning Re-engaged (1-24 hours)** | 1,260 | 8.4% | 46.51% | 51.34% | 0.6461 | 0.6610 |

### Bid Tier Slices (`cpo`)

| Slice | Count | % of Val | Actual CTR | Mean Pred | ROC AUC | Log Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **High CPO (>= 0.176)** | 7,503 | 50.02% | 30.67% | 31.79% | 0.6519 | 0.5846 |
| **Low CPO (< 0.176)** | 7,497 | 49.98% | 37.92% | 39.28% | 0.6752 | 0.6155 |

## 3. Discrimination vs. Calibration in Ad Systems

- **Ranking Discrimination** (`ROC AUC = 0.6689`): Reflects the model's ability to rank higher-intent ad candidates above lower-intent ones.
- **Probability Calibration** (`ECE = 0.0163`, `Brier = 0.2063`): Evaluates whether a predicted probability of 0.30 actually corresponds to a 30% click likelihood in production.
> *Why calibration is non-negotiable in advertising*: Ad auctions evaluate bids via $\text{eCPM} = \text{pCTR} \times \text{Bid}$. A model with great ROC AUC but inflated probabilities will cause the bidder to systematically overpay and exhaust marketing budgets.

