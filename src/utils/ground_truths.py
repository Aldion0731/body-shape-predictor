from dataclasses import dataclass

import pandas as pd

from .data_splits import SplitDatasets, TrainValTestDatasets


@dataclass
class SplitGroundTruths:
    bust: pd.Series
    hip: pd.Series
    waist: pd.Series


@dataclass
class TrainValTestGroundTruths:
    train: SplitGroundTruths
    val: SplitGroundTruths
    test: SplitGroundTruths


def get_ground_truth_values(
    train_val_test: TrainValTestDatasets,
) -> TrainValTestGroundTruths:
    return TrainValTestGroundTruths(
        train=get_ground_truth_split_values(train_val_test.train),
        val=get_ground_truth_split_values(train_val_test.val),
        test=get_ground_truth_split_values(train_val_test.test),
    )


def get_ground_truth_split_values(
    split_datasets: SplitDatasets,
) -> SplitGroundTruths:
    return SplitGroundTruths(
        bust=split_datasets.bust.targets,
        hip=split_datasets.hip.targets,
        waist=split_datasets.waist.targets,
    )
