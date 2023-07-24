import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

HIGHLIGHT_COLS = [
    "age",
    "gender",
    "height",
    "weight",
    "bust_errors_absolute",
    "hip_errors_absolute",
    "waist_errors_absolute",
]


class ErrorAnalyzer:
    def __init__(
        self,
        error_data: pd.DataFrame,
        column_num_categories_map: dict[str, int] = {},
        color_map: str = "jet",
    ) -> None:
        self.error_data = error_data
        self.error_data_with_categories = error_data.copy()
        self.map = column_num_categories_map
        self.color_map = color_map
        self.category_bins: dict[str, dict] = {}
        self.__categorize_data_columns()

    def __categorize_data_columns(self) -> None:
        for col in self.map.keys():
            self.__categorize_column(col, self.map[col])

    def __categorize_column(self, column: str, n_categories: int) -> None:
        column_data, intervals = pd.cut(
            self.error_data_with_categories[column], n_categories, retbins=True
        )
        self.category_bins[f"{column}"] = self.__get_bins(intervals)

        encoder = LabelEncoder()
        self.error_data_with_categories[column] = encoder.fit_transform(column_data)

    def get_category_means(
        self, category: str, highlight_cols: list[str] = HIGHLIGHT_COLS
    ) -> pd.DataFrame:
        error_data = self.error_data.copy()
        error_data[category] = self.error_data_with_categories[category]
        return self.__style(error_data.groupby(category).mean(), highlight_cols)

    def get_category_st_devs(
        self, category: str, highlight_cols: list[str] = HIGHLIGHT_COLS
    ) -> pd.DataFrame:
        error_data = self.error_data.copy()
        error_data[category] = self.error_data_with_categories[category]
        return self.__style(error_data.groupby(category).apply(np.std), highlight_cols)

    def get_category_counts(self, category: str) -> pd.Series:
        error_data = self.error_data.copy()
        error_data[category] = self.error_data_with_categories[category]
        return error_data.groupby(category).apply(lambda df: len(df))

    def filter_data_by_category(self, column: str, category_value: int) -> pd.DataFrame:
        mask = self.error_data_with_categories[column] == category_value
        return self.error_data[mask]

    def get_category_bin(self, category: str, category_value: int) -> tuple | None:
        category_ranges = self.category_bins.get(category)
        if category_ranges is None:
            return None
        return category_ranges.get(category_value)

    def __get_bins(self, intervals: np.ndarray) -> dict[int, tuple]:
        bins = [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)]
        return {i: bins[i] for i, _ in enumerate(bins)}

    def __style(self, df: pd.DataFrame, highlight_cols: list[str]) -> pd.DataFrame:
        highlight_cols = [col for col in highlight_cols if col in df.columns]
        return df.style.background_gradient(cmap=self.color_map, subset=highlight_cols)
