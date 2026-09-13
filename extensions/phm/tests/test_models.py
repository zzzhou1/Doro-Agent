import numpy as np
import torch
from phm_mcp.models import LSTMRegressor, TransformerRegressor, create_model
from phm_mcp.training import calculate_metrics


def test_lstm_and_transformer_output_one_value_per_window() -> None:
    values = torch.randn(4, 30, 17)
    assert LSTMRegressor(17)(values).shape == (4,)
    assert TransformerRegressor(17)(values).shape == (4,)


def test_model_factory_rejects_unknown_name() -> None:
    try:
        create_model("cnn", 17, 30, {})
    except ValueError as error:
        assert "Unknown model" in str(error)
    else:
        raise AssertionError("unknown model was accepted")


def test_metrics_are_zero_for_exact_predictions() -> None:
    metrics = calculate_metrics(np.asarray([1, 2]), np.asarray([1, 2]))
    assert metrics.rmse == 0
    assert metrics.mae == 0
    assert metrics.nasa_score == 0
