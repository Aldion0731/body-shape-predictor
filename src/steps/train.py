from src.utils.data_splits import SplitDatasets
from src.utils.models import FitParams, TrainHistory, VirtusizeModels
from src.utils.random_state import set_random_seed


def train(
    models: VirtusizeModels,
    train_datasets: SplitDatasets,
    fit_params: FitParams = FitParams(),
) -> TrainHistory:
    set_random_seed(0)

    print("Training bust circumference  prediction model...")
    bust_history = models.bust.fit(
        train_datasets.bust.features, train_datasets.bust.targets, **fit_params.bust
    )

    print("Training hip circumference prediction model...")
    hip_history = models.hip.fit(
        train_datasets.hip.features, train_datasets.hip.targets, **fit_params.hip
    )

    print("Training waist circumference prediction model...")
    waist_history = models.waist.fit(
        train_datasets.waist.features, train_datasets.waist.targets, **fit_params.waist
    )

    print("Training complete.")

    return TrainHistory(
        bust_history, hip_history, waist_history
    )  # useful for neural networks
