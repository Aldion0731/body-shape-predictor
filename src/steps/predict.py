import os

import pandas as pd

from src.utils.embed_predictions import embed_prediction_data
from src.utils.feature_engineer import FeatureEngineer
from src.utils.models import VirtusizeModels
from src.utils.predictions import get_split_predictions

from .load_data import load_config, load_serving_data


def predict_serving_data(
    models: VirtusizeModels,
    feature_engineers: list[FeatureEngineer] | None = None,
    predictions_file_name: str | None = None,
) -> pd.DataFrame:
    config = load_config()
    data_path = config.data_paths.test_data
    serving_data = load_serving_data(data_path, feature_engineers=feature_engineers)
    test_predictions = get_split_predictions(models, serving_data.processed_data)

    serving_data_with_predictions = embed_prediction_data(
        serving_data.original,
        serving_data.processed_data.bust.features,
        test_predictions,
    )

    if predictions_file_name is not None:
        preds_folder = config.data_paths.preds_folder
        os.makedirs(preds_folder, exist_ok=True)

        serving_data_with_predictions.to_csv(
            preds_folder / predictions_file_name, index=False
        )

    return serving_data_with_predictions
