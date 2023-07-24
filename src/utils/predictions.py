from dataclasses import dataclass

import numpy as np

from .data_splits import SplitDatasets, TrainValTestDatasets
from .models import ModelNames, VirtusizeModels


@dataclass
class SplitPredictions:
    bust: np.ndarray
    hip: np.ndarray
    waist: np.ndarray


@dataclass
class TrainValTestPredictions:
    train: SplitPredictions
    val: SplitPredictions
    test: SplitPredictions
    model_names: ModelNames


def get_train_val_test_predictions(
    models: VirtusizeModels, train_val_test: TrainValTestDatasets
) -> TrainValTestPredictions:
    return TrainValTestPredictions(
        train=get_split_predictions(models, train_val_test.train),
        val=get_split_predictions(models, train_val_test.val),
        test=get_split_predictions(models, train_val_test.test),
        model_names=models.names,
    )


def get_split_predictions(
    models: VirtusizeModels, split_datasets: SplitDatasets
) -> SplitPredictions:
    bust = models.bust.predict(split_datasets.bust.features)
    hip = models.hip.predict(split_datasets.hip.features)
    waist = models.waist.predict(split_datasets.waist.features)

    return SplitPredictions(
        bust=shape_predictions(bust),
        hip=shape_predictions(hip),
        waist=shape_predictions(waist),
    )


def shape_predictions(predictions: np.ndarray) -> np.ndarray:
    if len(predictions.shape) == 2:
        return predictions.reshape(
            predictions.shape[0]
        )  # Necessary for Neural Networks since predictions are returned as a 2D array
    return predictions
