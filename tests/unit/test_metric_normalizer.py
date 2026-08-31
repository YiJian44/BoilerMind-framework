from boilermind.experiment.metric_normalizer import normalize_metrics


def test_normalizes_mass_flow_metrics() -> None:
    result = normalize_metrics({
        "mae_t_h": 1.2,
        "rmse_t_h": 1.8,
        "r2": 0.91,
    })

    assert result == {
        "MAE": 1.2,
        "RMSE": 1.8,
        "R2": 0.91,
        "metric_unit": "t/h",
    }


def test_normalizes_volumetric_flow_metrics() -> None:
    result = normalize_metrics({
        "mae_m3_s": 0.12,
        "rmse_m3_s": 0.18,
        "r2_m3_s": 0.93,
    })

    assert result == {
        "MAE": 0.12,
        "RMSE": 0.18,
        "R2": 0.93,
        "metric_unit": "m3/s",
    }


def test_missing_r2_is_not_invented() -> None:
    result = normalize_metrics({
        "mae_m3_s": 0.12,
        "rmse_m3_s": 0.18,
    })

    assert result["MAE"] == 0.12
    assert result["RMSE"] == 0.18
    assert "R2" not in result


def test_existing_canonical_value_has_priority() -> None:
    result = normalize_metrics({"MAE": 3.0, "mae_t_h": 1.0})

    assert result["MAE"] == 3.0
