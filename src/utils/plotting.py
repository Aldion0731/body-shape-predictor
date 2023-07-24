import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def compare_error_distributions(
    error_data_1: pd.DataFrame,
    error_data2: pd.DataFrame,
    error_col: str,
    colors: tuple[str, str] = ("blue", "pink"),
    bins: int = 10,
) -> None:
    plt.hist(error_data_1[error_col], bins=bins, color=colors[0])
    plt.hist(error_data2[error_col], bins=10, alpha=0.5, color=colors[1])
    plt.title("Bust Error Distribution - Male(blue) vs Female(pink)")
    plt.show()


def compare_feature_distributions(
    feature: str, dataset_1: pd.DataFrame, dataset_2: pd.DataFrame
) -> Figure:
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(dataset_1[feature])
    plt.subplot(1, 2, 2)
    plt.hist(dataset_2[feature])
    return fig
