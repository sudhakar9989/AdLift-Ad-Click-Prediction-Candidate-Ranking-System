# AdLift Validation Model Comparison

Evaluated strictly on the chronological validation set (15,000 rows, 34.29% empirical click rate).

| Model | Log Loss (↓) | ROC AUC (↑) | PR AUC (↑) | Brier Score (↓) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ConstantBaseline** | 0.6437 | 0.5000 | 0.3429 | 0.2257 | 0.0s |
| **LogisticRegression** | 0.6026 | 0.6643 | 0.5153 | 0.2074 | 0.7256s |
| **LightGBM** | 0.6001 | 0.6689 | 0.5221 | 0.2063 | 1.0847s |
| **PyTorch_EmbeddingMLP** | 0.6070 | 0.6573 | 0.5065 | 0.2093 | 23.1512s |
