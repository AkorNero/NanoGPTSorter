from typing import Annotated, Literal  # noqa: F401

import yaml
from pydantic import BaseModel, Field  # noqa: F401


class DataConfig(BaseModel):
    vocab_size: Annotated[int, Field(gt=0)]
    special_characters: int
    min_seq: Annotated[int, Field(gt=0)]
    max_seq: Annotated[int, Field(gt=0)]


class ModelConfig(BaseModel):
    block_size: Annotated[int, Field(gt=0)]
    d_model: Annotated[int, Field(gt=0)]
    n_heads: Annotated[int, Field(gt=0)]
    n_layers: Annotated[int, Field(gt=0)]


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    epoch: Annotated[int, Field(ge=5)]
    agg_interval: Annotated[int, Field(ge=1, alias="agg_loss_interval")]
    device: Literal["cuda", "cpu"] = "cuda"


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def convert_yaml_to_dataclasses(path) -> Config:
    with open(path, "r") as f:
        config = yaml.safe_load(f.read())

    return Config(**config)


if __name__ == "__main__":
    config = convert_yaml_to_dataclasses("./configs/train_config.yaml")
    print(config)
    print(f"Loaded config successfully! Device: {config.training.device}")
    print(f"Vocab size: {config.data.vocab_size}")
