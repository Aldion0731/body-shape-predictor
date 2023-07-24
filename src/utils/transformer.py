from typing import Any, Protocol

import numpy as np


class Transformer(Protocol):
    def fit(self, vals_to_scale: Any) -> None:
        ...

    def transform(self, vals: Any) -> np.ndarray:
        ...

    def fit_transform(self, vals: Any) -> np.ndarray:
        ...
