"""
Model Registry — Experiment Tracking + Model Versioning
========================================================

Content-addressed model registry with 3-level entity model:
  training_run → prediction_set → backtest_run

Each entity is identified by a deterministic hash of its spec.
Enables reproducible experiments and skip-if-exists logic.

Source: ml4t-3e (review #18) — experiment tracking
        Orallexa (review #27) — content-addressed registry

Usage:
    from trading_modules.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Register a training run
    run_id = registry.register_training(
        model_name="xgboost_btc",
        params={"n_estimators": 200, "max_depth": 5},
        metrics={"accuracy": 0.65, "sharpe": 1.2},
    )

    # Check if already trained
    if registry.is_complete("xgboost_btc", params):
        print("Already trained — skip")
    else:
        # Train and register
        ...
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

@dataclass
class TrainingRun:
    """A model training run."""
    run_id: str
    model_name: str
    params_hash: str
    params: dict
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    status: str = "complete"  # complete / partial / failed

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """
    Content-addressed model registry.

    Every training run is identified by a hash of (model_name + params).
    This enables:
      - Skip-if-exists: don't retrain identical configs
      - Reproducibility: same params → same hash → same model
      - Audit trail: every run is tracked with metrics
    """

    def __init__(self, storage_path: str = "memory_data/model_registry.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, TrainingRun] = self._load()

    def register_training(
        self,
        model_name: str,
        params: dict,
        metrics: Optional[dict] = None,
        status: str = "complete",
    ) -> str:
        """Register a training run. Returns run_id."""
        params_hash = self._hash_params(model_name, params)
        run_id = f"{model_name}_{params_hash[:8]}"

        run = TrainingRun(
            run_id=run_id,
            model_name=model_name,
            params_hash=params_hash,
            params=params,
            metrics=metrics or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status,
        )

        self._runs[run_id] = run
        self._save()

        return run_id

    def is_complete(self, model_name: str, params: dict) -> bool:
        """Check if a training run with these params already exists."""
        params_hash = self._hash_params(model_name, params)
        run_id = f"{model_name}_{params_hash[:8]}"
        run = self._runs.get(run_id)
        return run is not None and run.status == "complete"

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a training run by ID."""
        return self._runs.get(run_id)

    def get_runs_by_model(self, model_name: str) -> list[TrainingRun]:
        """Get all runs for a model."""
        return [r for r in self._runs.values() if r.model_name == model_name]

    def get_best_run(self, model_name: str, metric: str = "sharpe") -> Optional[TrainingRun]:
        """Get best run for a model by metric."""
        runs = self.get_runs_by_model(model_name)
        if not runs:
            return None
        return max(runs, key=lambda r: r.metrics.get(metric, -999))

    def get_leaderboard(self, metric: str = "sharpe") -> list[dict]:
        """Get all runs sorted by metric."""
        runs = list(self._runs.values())
        runs.sort(key=lambda r: r.metrics.get(metric, -999), reverse=True)
        return [r.to_dict() for r in runs]

    def get_summary(self) -> dict:
        """Get registry summary."""
        models = set(r.model_name for r in self._runs.values())
        return {
            "total_runs": len(self._runs),
            "unique_models": len(models),
            "models": list(models),
            "best_per_model": {
                m: self.get_best_run(m).metrics if self.get_best_run(m) else None
                for m in models
            },
        }

    def _hash_params(self, model_name: str, params: dict) -> str:
        """Create deterministic hash of model + params."""
        spec = json.dumps({"model": model_name, "params": params}, sort_keys=True)
        return hashlib.sha256(spec.encode()).hexdigest()

    def _load(self) -> dict:
        """Major #20 fix: validate JSON structure before instantiating TrainingRun."""
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            # Validate each entry before instantiation
            valid_runs = {}
            required_fields = {"run_id", "model_name", "started_at"}
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                if not required_fields.issubset(v.keys()):
                    continue
                try:
                    valid_runs[k] = TrainingRun(**v)
                except (TypeError, ValueError):
                    continue
            return valid_runs
        except (json.JSONDecodeError, TypeError, OSError):
            return {}

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w") as f:
                json.dump(
                    {k: asdict(v) for k, v in self._runs.items()},
                    f, indent=2, default=str,
                )
        except OSError:
            pass
