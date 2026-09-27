import numpy as np
import pytest

from src.nonlinear_probe import fit_history_decoder
from scripts.run_replication import validate_configs


def test_decoder_fit_does_not_depend_on_test_features():
    rng = np.random.default_rng(33)
    x, y = rng.normal(size=(48, 4)), rng.normal(size=(48, 6))
    args = (x[:32], y[:32], x[32:40], y[32:40])
    a, fit_a, _ = fit_history_decoder(*args, x[40:], 1, epochs=2, every=1, weight_decays=(.001,), batch_size=16)
    b, fit_b, _ = fit_history_decoder(*args, x[40:]+100, 1, epochs=2, every=1, weight_decays=(.001,), batch_size=16)
    assert fit_a == fit_b
    assert not np.allclose(a, b)
    np.testing.assert_allclose(fit_a['input_scaler_mean'], x[:32].mean(0))
    np.testing.assert_allclose(fit_a['target_mean'], y[:32].mean(0))
    assert fit_a['test_used_for_selection'] is False


def test_replication_keeps_fixed_scientific_settings():
    configs = validate_configs()
    assert configs['treatment']['training']['seeds'] == [21,22,23,24,25]
    assert configs['treatment']['training']['gae_lambda'] == .95


def test_decoder_rejects_nonfinite_targets():
    x, y = np.ones((10, 4)), np.ones((10, 6))
    y[0,0] = np.nan
    with pytest.raises(ValueError, match='finite matrices'):
        fit_history_decoder(x, y, x, y, x, 1, epochs=2, every=1)
