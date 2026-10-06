from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration and metadata for one recommendation experiment."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty.")


@dataclass
class ExperimentResult:
    """Evaluation result produced by an experiment."""

    experiment_name: str
    metrics: dict[str, float]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("Experiment name cannot be empty.")

        for metric_name, value in self.metrics.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Metric '{metric_name}' must be numeric."
                )

    def as_dict(self) -> dict[str, Any]:
        """Return the experiment result as a serializable dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "parameters": dict(self.parameters),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create an experiment result from a dictionary."""
        return cls(
            experiment_name=data["experiment_name"],
            metrics=dict(data["metrics"]),
            parameters=dict(data.get("parameters", {})),
        )


class ExperimentRunner:
    """Track, compare, save, and load recommendation experiments."""

    def __init__(self) -> None:
        self.results: list[ExperimentResult] = []

    def record(
        self,
        config: ExperimentConfig,
        metrics: dict[str, float],
    ) -> ExperimentResult:
        """Record the result of an experiment."""
        result = ExperimentResult(
            experiment_name=config.name,
            metrics=dict(metrics),
            parameters=dict(config.parameters),
        )

        self.results.append(result)
        return result

    def get_results(self) -> list[ExperimentResult]:
        """Return all recorded experiment results."""
        return list(self.results)

    def best_result(
        self,
        metric: str,
        maximize: bool = True,
    ) -> ExperimentResult:
        """Return the best experiment according to one metric."""
        if not self.results:
            raise ValueError("No experiment results available.")

        for result in self.results:
            if metric not in result.metrics:
                raise ValueError(
                    f"Metric '{metric}' is missing from an experiment."
                )

        return (
            max(
                self.results,
                key=lambda result: result.metrics[metric],
            )
            if maximize
            else min(
                self.results,
                key=lambda result: result.metrics[metric],
            )
        )

    def save_results(self, path: str | Path) -> None:
        """Save all experiment results to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [result.as_dict() for result in self.results]

        output_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def load_results(self, path: str | Path) -> None:
        """Load experiment results from a JSON file."""
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Experiment results file not found: {input_path}"
            )

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        if not isinstance(data, list):
            raise ValueError(
                "Experiment results file must contain a JSON list."
            )

        self.results = [
            ExperimentResult.from_dict(item)
            for item in data
        ]

    def clear(self) -> None:
        """Remove all recorded experiment results."""
        self.results.clear()