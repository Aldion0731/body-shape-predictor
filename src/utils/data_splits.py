from dataclasses import dataclass
from enum import Enum

import pandas as pd


class TargetNames(str, Enum):
    BUST = "bust_circumference"
    HIP = "hip_circumference"
    WAIST = "waist_circumference"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class CsvSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class FeatureTargetPair:
    features: pd.DataFrame
    targets: pd.DataFrame


@dataclass
class SplitDatasets:
    bust: FeatureTargetPair
    hip: FeatureTargetPair
    waist: FeatureTargetPair


@dataclass
class TrainValTestDatasets:
    train: SplitDatasets
    val: SplitDatasets
    test: SplitDatasets


@dataclass
class ExperimentData:
    train_val_test: TrainValTestDatasets
    original_preprocessed: pd.DataFrame


@dataclass
class ServingData:
    processed_data: SplitDatasets
    original: pd.DataFrame


def make_train_val_test_datasets(csv_splits: CsvSplits) -> TrainValTestDatasets:
    train = make_datasets_from_df_split(csv_splits.train_df)
    val = make_datasets_from_df_split(csv_splits.val_df)
    test = make_datasets_from_df_split(csv_splits.test_df)

    return TrainValTestDatasets(train, val, test)


def make_datasets_from_df_split(df_split: pd.DataFrame) -> SplitDatasets:
    labels = [target.value for target in TargetNames]
    features = df_split.drop(labels, axis=1)
    bust = FeatureTargetPair(features=features, targets=df_split[TargetNames.BUST])
    hip = FeatureTargetPair(features=features, targets=df_split[TargetNames.HIP])
    waist = FeatureTargetPair(features=features, targets=df_split[TargetNames.WAIST])

    return SplitDatasets(bust, hip, waist)


def make_no_targets_dataset(data_no_targets: pd.DataFrame) -> SplitDatasets:
    bust = FeatureTargetPair(features=data_no_targets, targets=pd.Series())
    hip = FeatureTargetPair(features=data_no_targets, targets=pd.Series())
    waist = FeatureTargetPair(features=data_no_targets, targets=pd.Series())

    return SplitDatasets(bust, hip, waist)


def split_csv_data(
    data: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> CsvSplits:
    data = data.copy()
    data = data.sample(len(data))
    train_end_index = int(train_frac * len(data))
    val_end_index = train_end_index + int(val_frac * len(data))

    train_df = data.iloc[:train_end_index]
    val_df = data.iloc[train_end_index:val_end_index]
    test_df = data.iloc[val_end_index:]
    return CsvSplits(train_df, val_df, test_df)
