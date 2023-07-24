from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.data_splits import (
    CsvSplits,
    ExperimentData,
    ServingData,
    make_no_targets_dataset,
    make_train_val_test_datasets,
    split_csv_data,
)
from src.utils.feature_engineer import FeatureEngineer
from src.utils.preprocessor import preprocess_data
from src.utils.random_state import set_random_seed


def load_experiment_data(
    feature_engineers: list[FeatureEngineer] | None = None,
) -> ExperimentData:
    config = load_config()
    set_random_seed(config.random.seed)

    csv_data = pd.read_csv(config.data_paths.train_data)
    preprocessed_csv_data = preprocess_data(csv_data)
    original_preprocessed = preprocessed_csv_data.copy()

    split_params = config.split_params
    csv_splits = split_csv_data(
        preprocessed_csv_data,
        train_frac=split_params.train_frac,
        val_frac=split_params.val_frac,
    )

    if feature_engineers is not None:
        for transformer in feature_engineers:
            transformer.transform_features(csv_splits)

    train_val_test = make_train_val_test_datasets(csv_splits)

    return ExperimentData(train_val_test, original_preprocessed)


def load_serving_data(
    test_data_path: Path,
    feature_engineers: list[FeatureEngineer] | None = None,
) -> ServingData:
    csv_data = pd.read_csv(test_data_path)
    preprocessed_csv_data = preprocess_data(csv_data)

    placeholder_df = pd.DataFrame(columns=["gender", "age", "height", "weight"])
    csv_splits = CsvSplits(
        placeholder_df, placeholder_df.copy(), preprocessed_csv_data.copy()
    )

    if feature_engineers is not None:
        for transformer in feature_engineers:
            transformer.transform_features(csv_splits)

    processed_data = make_no_targets_dataset(csv_splits.test_df)

    return ServingData(processed_data=processed_data, original=csv_data)


# TODO: set the split sizes in a config file. These can be modified by modifying the Config object.
# TODO: Ensure there are no functions called within these functions that take keyword arguments that can't be modified.
