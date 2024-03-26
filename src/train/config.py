from enum import Enum

from pydantic import BaseModel


class DataSource(str, Enum):
    local = "local"
    network = "network"


class DatasetConfig(BaseModel):
    source: DataSource


class TrainConfig(BaseModel):
    epoch_num: int
    learning_rate: float
    batch_size: int

    weights_cache_path: str = "model_weights"

    training_metrics_cache_file: str = "training_metrics.csv"


class ValidationConfig(BaseModel):
    threshold: float


class ConfigSchema(BaseModel):
    dataset: DatasetConfig
    train: TrainConfig
    validation: ValidationConfig
