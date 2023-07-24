from dataclasses import dataclass
from pathlib import Path

from serde import Strict, serde
from serde.toml import from_toml


@serde(type_check=Strict)
@dataclass
class DataPaths:
    train_data: Path
    test_data: Path
    preds_folder: Path


@serde(type_check=Strict)
@dataclass
class Random:
    seed: int


@serde(type_check=Strict)
@dataclass
class SplitParams:
    train_frac: float
    val_frac: float


@serde(type_check=Strict)
@dataclass
class Config:
    data_paths: DataPaths
    random: Random
    split_params: SplitParams


def load_config(path: Path = Path("config.toml")) -> Config:
    with open(path) as f:
        return from_toml(Config, f.read())
