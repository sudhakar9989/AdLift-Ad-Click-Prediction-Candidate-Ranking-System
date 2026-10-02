"""Tests for compact PyTorch EmbeddingMLP click prediction model."""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from adlift.features import APPROVED_CATEGORICAL_FEATURES, APPROVED_INPUT_FEATURES
from adlift.models.pytorch_model import EmbeddingMLP, PyTorchClickModel


@pytest.fixture
def synthetic_train_val_data():
    """Create small synthetic train and validation datasets with approved features."""
    np.random.seed(42)
    n_train, n_val = 150, 50

    def _make_df(n):
        df = pd.DataFrame({
            "cpo": np.random.uniform(0.01, 1.0, size=n),
            "time_since_last_click": np.random.choice([-1, 100, 500, 2000], size=n),
            "click": np.random.binomial(1, 0.35, size=n),
        })
        for col in APPROVED_CATEGORICAL_FEATURES:
            df[col] = np.random.choice([101, 102, 103, 104, 105], size=n)
        return df

    return _make_df(n_train), _make_df(n_val)


def test_embedding_mlp_forward_pass():
    """Test raw EmbeddingMLP forward pass produces correct logits tensor."""
    batch_size = 16
    cat_cardinalities = [10, 5, 20, 8, 12, 15, 30, 10, 6, 7]  # 10 categorical features
    embedding_dims = [8, 8, 12, 8, 8, 8, 16, 8, 8, 8]
    num_continuous = 3

    model = EmbeddingMLP(
        cat_cardinalities=cat_cardinalities,
        embedding_dims=embedding_dims,
        num_continuous=num_continuous,
        hidden_dims=(32, 16),
    )

    num_x = torch.randn(batch_size, num_continuous)
    # Categorical indices in valid range [0, card]
    cat_x = torch.randint(0, 5, (batch_size, len(cat_cardinalities)))

    logits = model(num_x, cat_x)
    assert logits.shape == (batch_size,)
    assert not torch.isnan(logits).any()


def test_pytorch_click_model_fit_and_predict(synthetic_train_val_data):
    """Test PyTorchClickModel training, probability ranges, and unseen category resilience."""
    train_df, val_df = synthetic_train_val_data

    model = PyTorchClickModel(
        hidden_dims=(32, 16),
        batch_size=64,
        epochs=3,
        random_seed=42,
    )
    model.fit(train_df, val_df)

    # Introduce unseen categories in validation set
    val_dirty = val_df.copy()
    val_dirty.loc[0, "campaign"] = 9999999
    val_dirty.loc[1, "cat5"] = 8888888

    probs = model.predict_proba(val_dirty)
    assert probs.shape == (len(val_dirty), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-5)
    assert not np.isnan(probs).any(), "Probabilities must not contain NaNs!"


def test_pytorch_model_save_and_reload(synthetic_train_val_data, tmp_path):
    """Test saving PyTorch model weights and preprocessing checkpoint and reloading."""
    train_df, val_df = synthetic_train_val_data

    model = PyTorchClickModel(
        hidden_dims=(32, 16),
        batch_size=64,
        epochs=3,
        random_seed=42,
    )
    model.fit(train_df, val_df)

    checkpoint_path = str(tmp_path / "test_pt_model.pt")
    model.save(checkpoint_path)
    assert os.path.exists(checkpoint_path)

    reloaded_model = PyTorchClickModel.load(checkpoint_path)

    orig_probs = model.predict_proba(val_df)
    reloaded_probs = reloaded_model.predict_proba(val_df)

    np.testing.assert_allclose(orig_probs, reloaded_probs, atol=1e-5)
