import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.dropna(inplace=True)
    data["gender"] = (data["gender"] == "M").astype(int)

    cols_to_process = [
        "height",
        "bust_circumference",
        "waist_circumference",
        "hip_circumference",
    ]
    cols_to_process = [
        col for col in data.columns if col in cols_to_process
    ]  # Allows preprocessing even not all columns are present - in case of data without target values

    for col in cols_to_process:
        data[col] = data[col].replace(",", "", regex=True).astype(float)

    return data
