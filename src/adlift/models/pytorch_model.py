"""Compact PyTorch Embedding-MLP click prediction model for AdLift."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from adlift.features import (
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_INPUT_FEATURES,
    APPROVED_NUMERIC_FEATURES,
    validate_feature_columns,
)
from adlift.models.baselines import evaluate_predictions
from adlift.schema import TARGET_LABEL


class TabularClickDataset(Dataset):
    """PyTorch Dataset for numerical features and categorical token indices."""

    def __init__(
        self,
        num_tensor: torch.Tensor,
        cat_tensor: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_tensor = num_tensor
        self.cat_tensor = cat_tensor
        self.labels = labels

    def __len__(self) -> int:
        return len(self.num_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is not None:
            return self.num_tensor[idx], self.cat_tensor[idx], self.labels[idx]
        return self.num_tensor[idx], self.cat_tensor[idx]


class EmbeddingMLP(nn.Module):
    """Categorical embedding layers combined with continuous inputs feeding a lightweight MLP."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        embedding_dims: List[int],
        num_continuous: int = 3,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        # Embedding layer for each categorical feature (index 0 reserved for unknown/unseen)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 1, embedding_dim=dim, padding_idx=0)
            for card, dim in zip(cat_cardinalities, embedding_dims)
        ])

        total_embed_dim = sum(embedding_dims)
        in_dim = total_embed_dim + num_continuous

        # MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        embedded = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_rep = torch.cat(embedded, dim=1)
        x = torch.cat([num_x, cat_rep], dim=1)
        logits = self.mlp(x)
        return logits.squeeze(-1)


class PyTorchClickModel:
    """Wrapper providing fit, predict_proba, and save/load for EmbeddingMLP."""

    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (64, 32),
        batch_size: int = 1024,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        epochs: int = 8,
        random_seed: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cat_cols = list(APPROVED_CATEGORICAL_FEATURES)
        self.num_cols = list(APPROVED_NUMERIC_FEATURES)

        # Preprocessing artifacts fitted strictly on training data
        self.vocabularies_: Dict[str, Dict[Any, int]] = {}
        self.cat_cardinalities_: List[int] = []
        self.embedding_dims_: List[int] = []
        self.num_stats_: Dict[str, Dict[str, float]] = {}

        self.model_: Optional[EmbeddingMLP] = None

    def _fit_preprocessor(self, train_df: pd.DataFrame) -> None:
        """Build vocabularies and continuous scalers strictly using training data."""
        # 1. Numeric statistics
        cpo_median = float(train_df["cpo"].median())
        cpo_clean = train_df["cpo"].fillna(cpo_median)
        self.num_stats_["cpo"] = {
            "median": cpo_median,
            "mean": float(cpo_clean.mean()),
            "std": float(cpo_clean.std() + 1e-7),
        }

        ts = train_df["time_since_last_click"]
        valid_ts = ts[ts >= 0]
        ts_median = float(valid_ts.median()) if len(valid_ts) > 0 else 0.0
        ts_clean = ts.replace(-1, ts_median)
        self.num_stats_["time_since_last_click"] = {
            "median": ts_median,
            "mean": float(ts_clean.mean()),
            "std": float(ts_clean.std() + 1e-7),
        }

        # 2. Categorical vocabularies
        self.vocabularies_ = {}
        self.cat_cardinalities_ = []
        self.embedding_dims_ = []

        for col in self.cat_cols:
            unique_vals = train_df[col].dropna().unique()
            vocab = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.vocabularies_[col] = vocab
            card = len(vocab)
            self.cat_cardinalities_.append(card)
            # Embedding dimension heuristic: min(32, max(8, int(card^0.25 * 8)))
            dim = min(32, max(8, int((card ** 0.25) * 8)))
            self.embedding_dims_.append(dim)

    def _transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform raw input DataFrame into PyTorch tensors using fitted preprocessing."""
        # Numeric transformation (3 features: cpo_scaled, ts_scaled, ts_missing_indicator)
        cpo = df["cpo"].fillna(self.num_stats_["cpo"]["median"]).values
        cpo_norm = (cpo - self.num_stats_["cpo"]["mean"]) / self.num_stats_["cpo"]["std"]

        ts = df["time_since_last_click"].values
        ts_indicator = (ts == -1).astype(float)
        ts_imputed = np.where(ts == -1, self.num_stats_["time_since_last_click"]["median"], ts)
        ts_norm = (ts_imputed - self.num_stats_["time_since_last_click"]["mean"]) / self.num_stats_["time_since_last_click"]["std"]

        num_arr = np.column_stack([cpo_norm, ts_norm, ts_indicator]).astype(np.float32)

        # Categorical transformation: maps unknown hashes to 0 (<UNK>)
        cat_arrs = []
        for col in self.cat_cols:
            vocab = self.vocabularies_[col]
            indices = df[col].map(lambda x: vocab.get(x, 0)).values.astype(np.int64)
            cat_arrs.append(indices)

        cat_arr = np.column_stack(cat_arrs)

        return torch.tensor(num_arr, dtype=torch.float32), torch.tensor(cat_arr, dtype=torch.long)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "PyTorchClickModel":
        """Fit preprocessing on train_df and train the EmbeddingMLP with early stopping on validation."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self._fit_preprocessor(train_df)

        num_train, cat_train = self._transform(train_df)
        y_train = torch.tensor(train_df[TARGET_LABEL].values, dtype=torch.float32)

        train_dataset = TabularClickDataset(num_train, cat_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if val_df is not None:
            num_val, cat_val = self._transform(val_df)
            y_val = torch.tensor(val_df[TARGET_LABEL].values, dtype=torch.float32)
            val_dataset = TabularClickDataset(num_val, cat_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)
        else:
            val_loader = None

        self.model_ = EmbeddingMLP(
            cat_cardinalities=self.cat_cardinalities_,
            embedding_dims=self.embedding_dims_,
            num_continuous=3,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        patience = 2
        patience_counter = 0
        best_state = None

        print(f"Training PyTorch EmbeddingMLP on {self.device} (total epochs: {self.epochs})...")
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            total_loss = 0.0
            batches = 0

            for num_b, cat_b, y_b in train_loader:
                num_b = num_b.to(self.device)
                cat_b = cat_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(num_b, cat_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_train_loss = total_loss / batches

            # Validation step for early stopping
            if val_loader is not None:
                self.model_.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for num_v, cat_v, y_v in val_loader:
                        num_v = num_v.to(self.device)
                        cat_v = cat_v.to(self.device)
                        y_v = y_v.to(self.device)
                        val_logits = self.model_(num_v, cat_v)
                        val_loss_sum += criterion(val_logits, y_v).item()
                        val_batches += 1
                avg_val_loss = val_loss_sum / val_batches
                print(f"  Epoch {epoch:2d}/{self.epochs:2d} | Train BCE: {avg_train_loss:.4f} | Val BCE: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss - 1e-4:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping triggered at epoch {epoch} (best val BCE: {best_val_loss:.4f})")
                        break
            else:
                print(f"  Epoch {epoch:2d}/{self.epochs:2d} | Train BCE: {avg_train_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict calibrated click probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")

        self.model_.eval()
        num_x, cat_x = self._transform(df)
        dataset = TabularClickDataset(num_x, cat_x)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for num_b, cat_b in loader:
                num_b = num_b.to(self.device)
                cat_b = cat_b.to(self.device)
                logits = self.model_(num_b, cat_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        p1 = np.concatenate(all_probs, axis=0).reshape(-1, 1)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)

    def save(self, filepath: str) -> None:
        """Save model weights, configuration, and fitted preprocessing dictionaries."""
        if self.model_ is None:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_state": self.model_.state_dict(),
            "cat_cardinalities": self.cat_cardinalities_,
            "embedding_dims": self.embedding_dims_,
            "vocabularies": self.vocabularies_,
            "num_stats": self.num_stats_,
            "hidden_dims": self.hidden_dims,
            "random_seed": self.random_seed,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PyTorchClickModel":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        instance = cls(
            hidden_dims=checkpoint["hidden_dims"],
            random_seed=checkpoint["random_seed"],
        )
        instance.cat_cardinalities_ = checkpoint["cat_cardinalities"]
        instance.embedding_dims_ = checkpoint["embedding_dims"]
        instance.vocabularies_ = checkpoint["vocabularies"]
        instance.num_stats_ = checkpoint["num_stats"]

        instance.model_ = EmbeddingMLP(
            cat_cardinalities=instance.cat_cardinalities_,
            embedding_dims=instance.embedding_dims_,
            num_continuous=3,
            hidden_dims=instance.hidden_dims,
        )
        instance.model_.load_state_dict(checkpoint["model_state"])
        instance.model_.to(instance.device)
        instance.model_.eval()
        return instance


def train_and_evaluate_pytorch(
    train_path: str = os.path.join("data", "processed", "train.csv"),
    val_path: str = os.path.join("data", "processed", "validation.csv"),
    models_dir: str = "models",
    comparison_report_path: str = os.path.join("reports", "model_comparison.json"),
    comparison_md_path: str = os.path.join("reports", "model_comparison.md"),
) -> Dict[str, Any]:
    """Train PyTorch EmbeddingMLP, evaluate on validation split, and update comparison table."""
    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading validation data: {val_path}")
    val_df = pd.read_csv(val_path)

    validate_feature_columns(APPROVED_INPUT_FEATURES)

    print("\n--- Training PyTorch Categorical Embedding-MLP ---")
    t0 = time.time()
    pt_model = PyTorchClickModel(
        hidden_dims=(64, 32),
        batch_size=1024,
        learning_rate=0.003,
        epochs=8,
        random_seed=42,
    )
    pt_model.fit(train_df, val_df)
    train_time = time.time() - t0

    # Evaluate on validation split
    val_probs = pt_model.predict_proba(val_df)[:, 1]
    y_val = val_df[TARGET_LABEL].values
    metrics = evaluate_predictions(y_val, val_probs)
    metrics["train_time_seconds"] = round(train_time, 4)
    metrics["architecture"] = "EmbeddingMLP (64 -> 32 -> 1)"

    # Save model artifact
    os.makedirs(models_dir, exist_ok=True)
    pt_artifact_path = os.path.join(models_dir, "pytorch_click_model.pt")
    pt_model.save(pt_artifact_path)
    print(f"Saved PyTorch model checkpoint to: {pt_artifact_path}")

    # Load existing comparison table and update
    comparison_table: Dict[str, Any] = {
        "validation_rows": len(val_df),
        "target_label": TARGET_LABEL,
        "models": {},
    }
    if os.path.exists(comparison_report_path):
        with open(comparison_report_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            comparison_table["models"].update(existing.get("models", {}))

    comparison_table["models"]["PyTorch_EmbeddingMLP"] = metrics

    # Save updated comparison JSON
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_table, f, indent=2)
    print(f"Updated comparison JSON report at: {comparison_report_path}")

    # Save updated Markdown comparison table
    md_content = "# AdLift Validation Model Comparison\n\n"
    md_content += "Evaluated strictly on the chronological validation set (15,000 rows, 34.29% empirical click rate).\n\n"
    md_content += "| Model | Log Loss (↓) | ROC AUC (↑) | PR AUC (↑) | Brier Score (↓) | Train Time (s) |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for name, m in comparison_table["models"].items():
        md_content += f"| **{name}** | {m['log_loss']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['brier_score']:.4f} | {m['train_time_seconds']}s |\n"

    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Updated Markdown comparison table at: {comparison_md_path}")

    return comparison_table
