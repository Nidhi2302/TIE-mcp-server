"""
Minimal in-memory ModelManager stub.

Provides the interface expected by:
- src/tie_mcp/core/engine_manager.py
- tests/conftest.py (AsyncMock(spec=ModelManager))
- server handlers that may rely on model listing

This is a lightweight, non-persistent implementation intended to
unblock tests by satisfying import and attribute expectations.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from ..core.tie.engine import TechniqueInferenceEngine


@dataclass
class _ModelRecord:
    id: str
    name: str
    model_type: str
    status: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    dataset_path: str
    artifacts_path: str
    description: str
    version: str
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelManager:
    """
    In-memory model registry.

    Methods implemented to match usage patterns:
    - save_model(...) -> str
    - load_model(model_id) -> TechniqueInferenceEngine
    - get_default_model() -> _ModelRecord | None
    - list_models(...)
    - get_model_info(model_id)
    - delete_model(model_id)
    - create_dataset(...) (stub)
    - set_default_model(model_id)
    - initialize()/cleanup()
    """

    def __init__(self) -> None:
        self._models: dict[str, _ModelRecord] = {}
        self._engines: dict[str, TechniqueInferenceEngine] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        async with self._lock:
            self._models.clear()
            self._engines.clear()
        self._initialized = False

    async def save_model(
        self,
        engine: TechniqueInferenceEngine,
        model_type: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
        dataset_path: str,
        description: str = "TIE model",
        artifacts_path: str | None = None,
        version: str = "1.0.0",
        set_default: bool = True,
    ) -> str:
        """
        Store engine + metadata, return model_id.
        """
        async with self._lock:
            model_id = str(uuid.uuid4())
            record = _ModelRecord(
                id=model_id,
                name=f"{model_type}-{model_id[:8]}",
                model_type=model_type,
                status="trained",
                hyperparameters=hyperparameters,
                metrics=metrics,
                dataset_path=dataset_path,
                artifacts_path=artifacts_path or "",
                description=description,
                version=version,
                is_default=False,
            )
            self._models[model_id] = record
            self._engines[model_id] = engine
            if set_default and self.get_default_model_sync() is None:
                record.is_default = True
            return model_id

    async def load_model(self, model_id: str) -> TechniqueInferenceEngine:
        async with self._lock:
            if model_id not in self._engines:
                raise ValueError(f"Model {model_id} not found")
            return self._engines[model_id]

    def get_default_model_sync(self) -> _ModelRecord | None:
        for rec in self._models.values():
            if rec.is_default:
                return rec
        return None

    async def get_default_model(self) -> _ModelRecord | None:
        return self.get_default_model_sync()

    async def list_models(self, status: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            models = list(self._models.values())
            if status:
                models = [m for m in models if m.status == status]
            return [m.to_dict() for m in models]

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._models.get(model_id)
            if record is None:
                raise ValueError(f"Model {model_id} not found")
            return record.to_dict()

    async def delete_model(self, model_id: str) -> None:
        async with self._lock:
            if model_id in self._models:
                was_default = self._models[model_id].is_default
                del self._models[model_id]
                self._engines.pop(model_id, None)
                if was_default:
                    # Promote any remaining model to default
                    for rec in self._models.values():
                        rec.is_default = True
                        break
            else:
                raise ValueError(f"Model {model_id} not found")

    async def set_default_model(self, model_id: str) -> None:
        async with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")
            for rec in self._models.values():
                rec.is_default = False
            self._models[model_id].is_default = True

    async def create_dataset(
        self,
        raw_data_path: str,
        output_dataset_path: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Stub dataset creation: just returns metadata.
        """
        return {
            "dataset_id": str(uuid.uuid4()),
            "source": raw_data_path,
            "path": output_dataset_path,
            "description": description or "Dataset (stub)",
        }


__all__ = ["ModelManager"]
