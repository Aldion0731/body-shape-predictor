from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .data_splits import TrainValTestDatasets


class Predictor(Protocol):
    def fit(self, X: Any, y: Any, **kwargs: dict[str, Any]) -> Any:
        ...

    def predict(self, y: Any) -> np.ndarray:
        ...


@dataclass
class ModelNames:
    bust: str
    hip: str
    waist: str


@dataclass
class VirtusizeModels:
    bust: Predictor
    hip: Predictor
    waist: Predictor
    names: ModelNames


@dataclass
class FitParams:
    bust: dict[str, Any] = field(default_factory=lambda: {})
    hip: dict[str, Any] = field(default_factory=lambda: {})
    waist: dict[str, Any] = field(default_factory=lambda: {})

    def update_with_validation_data(self, datasplits: TrainValTestDatasets) -> None:
        bust = (datasplits.val.bust.features, datasplits.val.bust.targets)
        hip = (datasplits.val.hip.features, datasplits.val.hip.targets)
        waist = (datasplits.val.waist.features, datasplits.val.waist.targets)

        self.bust.update({"validation_data": bust})
        self.hip.update({"validation_data": hip})
        self.waist.update({"validation_data": waist})

    def update_all(self, params_update: dict[str, Any]) -> None:
        self.bust.update(params_update)
        self.hip.update(params_update)
        self.waist.update(params_update)


@dataclass
class TrainHistory:
    bust: Any
    hip: Any
    waist: Any
