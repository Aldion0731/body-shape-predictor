from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data_splits import SplitType


class Transformer(Protocol):
    def fit(self, vals_to_scale: Any) -> None:
        ...

    def transform(self, vals: Any) -> np.ndarray:
        ...

    def fit_transform(self, vals: Any) -> np.ndarray:
        ...


def transform_split_features(
    transformer: Transformer,
    data: pd.DataFrame,
    split_type: SplitType,
    cols_to_transform: list[str],
) -> None:
    if split_type == SplitType.TRAIN:
        data[cols_to_transform] = transformer.fit_transform(data[cols_to_transform])
    else:
        data[cols_to_transform] = transformer.transform(data[cols_to_transform])
