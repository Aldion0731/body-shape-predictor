from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .ground_truths import SplitGroundTruths, TrainValTestGroundTruths
from .models import ModelNames
from .predictions import SplitPredictions, TrainValTestPredictions


@dataclass
class SplitScores:
    bust: float
    hip: float
    waist: float


@dataclass
class TrainValTestScores:
    train: SplitScores
    val: SplitScores
    test: SplitScores


@dataclass
class Metrics:
    mae: TrainValTestScores
    mse: TrainValTestScores
    r2: TrainValTestScores


class MetricNames(Enum):
    mae = "mae"
    mse = "mse"
    r2 = "r2"


class MetricsManager:
    def __init__(self) -> None:
        self.metrics: list[Metrics] = []
        self.model_names: list[ModelNames] = []
        self.metrics_table = pd.DataFrame([])

    def update_metrics(
        self,
        ground_truths: TrainValTestGroundTruths,
        predictions: TrainValTestPredictions,
    ) -> pd.DataFrame:
        mae = compute_train_val_test_metrics(
            mean_absolute_error, ground_truths, predictions
        )
        mse = compute_train_val_test_metrics(
            mean_squared_error, ground_truths, predictions
        )
        r2 = compute_train_val_test_metrics(r2_score, ground_truths, predictions)
        self.metrics.append(Metrics(mae, mse, r2))
        self.model_names.append(predictions.model_names)

        all_metric_dfs: list[pd.DataFrame] = []

        for metric in MetricNames:
            train_test_val_scores: TrainValTestScores = getattr(
                self.metrics[-1], metric.value
            )
            metric_df = self.get_single_metric_df(train_test_val_scores, metric)
            all_metric_dfs.append(metric_df)

        table_update = pd.concat(all_metric_dfs, axis=1)
        self.metrics_table = pd.concat([self.metrics_table, table_update])
        return self.metrics_table

    def get_single_metric_df(
        self, train_test_val_scores: TrainValTestScores, metric_name: MetricNames
    ) -> pd.DataFrame:
        metric_df = pd.DataFrame(asdict(train_test_val_scores))
        metric_df.columns = [col + f"_{metric_name.value}" for col in metric_df.columns]

        model_names = self.model_names[-1]
        if list(metric_df.index) == ["bust", "hip", "waist"]:
            metric_df.index = [model_names.bust, model_names.hip, model_names.waist]

        return metric_df

    def show_validation_metrics(self) -> pd.DataFrame:
        validation_columns = [
            col for col in self.metrics_table.columns if col.startswith("val")
        ]
        return self.metrics_table.loc[:, validation_columns]


def compute_train_val_test_metrics(
    score_func: Callable[[Sequence, np.ndarray], float],
    gound_truths: TrainValTestGroundTruths,
    predictions: TrainValTestPredictions,
) -> TrainValTestScores:
    train = compute_split_metrics(score_func, gound_truths.train, predictions.train)
    val = compute_split_metrics(score_func, gound_truths.val, predictions.val)
    test = compute_split_metrics(score_func, gound_truths.test, predictions.test)
    return TrainValTestScores(train, val, test)


def compute_split_metrics(
    score_func: Callable[[Sequence, np.ndarray], float],
    split_ground_truths: SplitGroundTruths,
    split_predictions: SplitPredictions,
) -> SplitScores:
    bust = score_func(split_ground_truths.bust, split_predictions.bust)
    hip = score_func(split_ground_truths.hip, split_predictions.hip)
    waist = score_func(split_ground_truths.waist, split_predictions.waist)

    return SplitScores(bust, hip, waist)
