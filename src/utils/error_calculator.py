from dataclasses import dataclass

import pandas as pd

from .ground_truths import SplitGroundTruths
from .predictions import SplitPredictions


@dataclass
class SplitErrors:
    bust: pd.Series
    hip: pd.Series
    waist: pd.Series


def get_split_error_data(
    ground_truths: SplitGroundTruths,
    predictions: SplitPredictions,
    original_data: pd.DataFrame,
) -> pd.DataFrame:
    split_errors = calculate_split_errors(ground_truths, predictions)
    errors_data = original_data.copy()

    errors_data["bust_errors"] = split_errors.bust
    errors_data["hip_errors"] = split_errors.hip
    errors_data["waist_errors"] = split_errors.waist

    errors_data["bust_errors_absolute"] = split_errors.bust.apply(abs)
    errors_data["hip_errors_absolute"] = split_errors.hip.apply(abs)
    errors_data["waist_errors_absolute"] = split_errors.waist.apply(abs)

    return errors_data.dropna()  # Drops the 2 other splits not being used.


def calculate_split_errors(
    ground_truths: SplitGroundTruths, predictions: SplitPredictions
) -> SplitErrors:
    bust_errors: pd.Series = predictions.bust - ground_truths.bust
    hip_errors: pd.Series = predictions.hip - ground_truths.hip
    waist_errors: pd.Series = predictions.waist - ground_truths.waist

    return SplitErrors(bust_errors, hip_errors, waist_errors)
