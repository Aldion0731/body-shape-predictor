import pandas as pd

from src.utils.data_splits import TargetNames
from src.utils.predictions import SplitPredictions


def embed_prediction_data(
    original_unprocessed_data: pd.DataFrame,
    processed_features: pd.DataFrame,
    predictions: SplitPredictions,
) -> pd.DataFrame:
    mapped_data = map_targets_to_features(processed_features, predictions)

    for target in TargetNames:
        original_unprocessed_data[target.value] = mapped_data[target.value]
        original_unprocessed_data[target.value].fillna(
            mapped_data[target.value].mean(), inplace=True
        )

    return original_unprocessed_data


def map_targets_to_features(
    features: pd.DataFrame, predictions: SplitPredictions
) -> pd.DataFrame:
    features = features.copy()

    for target in TargetNames:
        features[target.value] = getattr(predictions, target.name.lower())

    return features
