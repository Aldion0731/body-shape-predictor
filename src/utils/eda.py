import matplotlib.pyplot as plt
import pandas as pd


def get_train_test_feature_comparison_table(
    train_data: pd.DataFrame, test_data: pd.DataFrame, feature: str
) -> pd.DataFrame:
    data = {
        "mean": [train_data[feature].mean(), test_data[feature].mean()],
        "st_dev": [train_data[feature].std(), test_data[feature].std()],
        "max": [train_data[feature].max(), test_data[feature].max()],
        "min": [train_data[feature].min(), test_data[feature].min()],
        "count": [train_data[feature].shape[0], test_data[feature].shape[0]],
    }

    return pd.DataFrame(data, index=["train_data", "test_data"])


def display_train_test_feature_comparison_histograms(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature: str,
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.hist(train_data[feature], axes=ax1)
    ax2.hist(test_data[feature], axes=ax2)
    plt.suptitle(
        f"Train ({len(train_data)} examples ) vs Test ({len(test_data)} examples) '{feature}' Distributions ()"
    )
    plt.tight_layout()
