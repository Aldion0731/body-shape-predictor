from typing import Any

from keras import Sequential
from keras.layers import Dense, Dropout


def build_model(input_dim: int, compile_params: dict[str, Any] = {}) -> Sequential:
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", **compile_params)

    return model
