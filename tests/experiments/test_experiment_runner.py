import pytest

from src.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)


def test_experiment_config():
    config = ExperimentConfig(
        name="baseline",
        parameters={"learning_rate": 0.001},
    )

    assert config.name == "baseline"
    assert config.parameters["learning_rate"] == 0.001


def test_experiment_config_rejects_empty_name():
    with pytest.raises(ValueError):
        ExperimentConfig(name="")


def test_experiment_result():
    result = ExperimentResult(
        experiment_name="baseline",
        metrics={"precision": 0.8, "ndcg": 0.7},
    )

    assert result.experiment_name == "baseline"
    assert result.metrics["precision"] == 0.8


def test_experiment_result_as_dict():
    result = ExperimentResult(
        experiment_name="baseline",
        metrics={"ndcg": 0.75},
        parameters={"k": 10},
    )

    data = result.as_dict()

    assert data["experiment_name"] == "baseline"
    assert data["metrics"]["ndcg"] == 0.75
    assert data["parameters"]["k"] == 10


def test_experiment_result_rejects_non_numeric_metric():
    with pytest.raises(TypeError):
        ExperimentResult(
            experiment_name="baseline",
            metrics={"ndcg": "invalid"},
        )


def test_runner_starts_empty():
    runner = ExperimentRunner()

    assert runner.get_results() == []


def test_runner_records_result():
    runner = ExperimentRunner()

    config = ExperimentConfig(
        name="baseline",
        parameters={"epochs": 5},
    )

    result = runner.record(
        config,
        {"precision": 0.8, "ndcg": 0.7},
    )

    assert result.experiment_name == "baseline"
    assert len(runner.get_results()) == 1


def test_runner_records_multiple_experiments():
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(name="experiment_a"),
        {"ndcg": 0.70},
    )

    runner.record(
        ExperimentConfig(name="experiment_b"),
        {"ndcg": 0.82},
    )

    assert len(runner.get_results()) == 2


def test_best_result_maximizes_metric():
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(name="experiment_a"),
        {"ndcg": 0.70},
    )

    runner.record(
        ExperimentConfig(name="experiment_b"),
        {"ndcg": 0.82},
    )

    best = runner.best_result("ndcg")

    assert best.experiment_name == "experiment_b"


def test_best_result_minimizes_metric():
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(name="experiment_a"),
        {"loss": 0.30},
    )

    runner.record(
        ExperimentConfig(name="experiment_b"),
        {"loss": 0.20},
    )

    best = runner.best_result("loss", maximize=False)

    assert best.experiment_name == "experiment_b"


def test_best_result_rejects_empty_runner():
    runner = ExperimentRunner()

    with pytest.raises(ValueError):
        runner.best_result("ndcg")


def test_best_result_rejects_missing_metric():
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(name="baseline"),
        {"precision": 0.8},
    )

    with pytest.raises(ValueError):
        runner.best_result("ndcg")


def test_clear_removes_results():
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(name="baseline"),
        {"ndcg": 0.75},
    )

    runner.clear()

    assert runner.get_results() == []


def test_experiment_result_from_dict():
    data = {
        "experiment_name": "baseline",
        "metrics": {"ndcg": 0.75},
        "parameters": {"k": 10},
    }

    result = ExperimentResult.from_dict(data)

    assert result.experiment_name == "baseline"
    assert result.metrics["ndcg"] == 0.75
    assert result.parameters["k"] == 10


def test_runner_save_results(tmp_path):
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(
            name="baseline",
            parameters={"epochs": 5},
        ),
        {"precision": 0.8, "ndcg": 0.7},
    )

    output_path = tmp_path / "experiments.json"

    runner.save_results(output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8")


def test_runner_load_results(tmp_path):
    runner = ExperimentRunner()

    runner.record(
        ExperimentConfig(
            name="baseline",
            parameters={"epochs": 5},
        ),
        {"precision": 0.8, "ndcg": 0.7},
    )

    output_path = tmp_path / "experiments.json"

    runner.save_results(output_path)

    loaded_runner = ExperimentRunner()
    loaded_runner.load_results(output_path)

    results = loaded_runner.get_results()

    assert len(results) == 1
    assert results[0].experiment_name == "baseline"
    assert results[0].metrics["ndcg"] == 0.7
    assert results[0].parameters["epochs"] == 5


def test_runner_load_results_rejects_missing_file(tmp_path):
    runner = ExperimentRunner()

    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        runner.load_results(missing_path)


def test_runner_load_results_rejects_invalid_format(tmp_path):
    runner = ExperimentRunner()

    invalid_path = tmp_path / "invalid.json"

    invalid_path.write_text(
        '{"experiment_name": "baseline"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        runner.load_results(invalid_path)