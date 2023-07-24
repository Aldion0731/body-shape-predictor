from abc import abstractmethod

import pandas as pd

from .data_splits import CsvSplits, SplitType
from .transformer import Transformer


class FeatureEngineer:
    @abstractmethod
    def transform_features(self, csv_splits: CsvSplits) -> None:
        pass


class FeatureScaler(FeatureEngineer):
    def __init__(self, scaler: Transformer, cols_to_scale: list[str]) -> None:
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale

    def transform_features(self, csv_splits: CsvSplits) -> None:
        self.scale_split_features(csv_splits.train_df, SplitType.TRAIN)
        self.scale_split_features(csv_splits.val_df, SplitType.VAL)
        self.scale_split_features(csv_splits.test_df, SplitType.TEST)

    def scale_split_features(self, data: pd.DataFrame, split_type: SplitType) -> None:
        if len(data) == 0:
            print(f"No data for {split_type.value} split so split was not scaled.")
            return

        if split_type == SplitType.TRAIN:
            data[self.cols_to_scale] = self.scaler.fit_transform(
                data[self.cols_to_scale]
            )
        else:
            data[self.cols_to_scale] = self.scaler.transform(data[self.cols_to_scale])


class BMIProxyEngineer(FeatureEngineer):
    def transform_features(self, csv_splits: CsvSplits) -> None:
        train_df, val_df, test_df = (
            csv_splits.train_df,
            csv_splits.val_df,
            csv_splits.test_df,
        )

        train_df["bmi_proxy"] = train_df["height"] / train_df["weight"]
        val_df["bmi_proxy"] = val_df["height"] / val_df["weight"]
        test_df["bmi_proxy"] = test_df["height"] / test_df["weight"]


class FeatureDropper(FeatureEngineer):
    def __init__(self, cols_to_drop: list[str]):
        self.cols_to_drop = cols_to_drop

    def transform_features(self, csv_splits: CsvSplits) -> None:
        csv_splits.train_df.drop(self.cols_to_drop, axis=1, inplace=True)
        csv_splits.val_df.drop(self.cols_to_drop, axis=1, inplace=True)
        csv_splits.test_df.drop(self.cols_to_drop, axis=1, inplace=True)
