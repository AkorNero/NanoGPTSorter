import random

import mlflow
import mlflow.pytorch
import torch
from dotenv import load_dotenv  # noqa: F401
from torch.utils.data import DataLoader

from data import vocab
from data.dataset import SorterDataset
from model import config
from model.transformer import NanoGPTSorter


def setup():
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("NanoGPTSorter")


def single_epoch(epoch_index: int, agg_interval: int, dataloader: DataLoader):
    running_loss: float = 0.0
    average_loss: float = 0.0
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(
            device
        )  # unlike nn.Module objects the tesor.to() method returns a new tensor
        Y = Y.to(device)
        optimizer.zero_grad()
        prediction, loss = model(X, target=Y)
        loss.backward()
        optimizer.step()

        log_idx = epoch_index * len(dataloader) + i + 1

        mlflow.log_metric("train_loss", loss.item(), step=log_idx)

        running_loss += loss.item()
        if (i + 1) % agg_interval == 0:
            average_loss = running_loss / agg_interval

            print(f"\n\tbatch {i + 1} loss: {average_loss}")
            mlflow.log_metric(
                f"average_training_loss_per_{agg_interval}_batch",
                average_loss,
                step=log_idx,
            )

            running_loss = 0.0
    mlflow.log_metric("epoch_train_loss", average_loss)
    return average_loss


setup()

device = (
    "cuda"
    if (config.training.device == "cuda" and torch.cuda.is_available())
    else "cpu"
)

sequences = set()
while len(sequences) < 6000:
    seq_length = random.randint(1, 5)
    seq = tuple(
        random.choices(vocab, k=seq_length)
    )  # can not add unhasable list to a set
    sequences.add(seq)

sequences = [list(s) for s in sequences]
random.shuffle(sequences)

n = len(sequences)
train_loader = DataLoader(
    SorterDataset(sequences[: int(n * 0.8)], config.model.block_size),
    batch_size=config.training.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    SorterDataset(sequences[int(n * 0.8) : int(n * 0.9)], config.model.block_size),
    batch_size=config.training.batch_size,
)
test_loader = DataLoader(
    SorterDataset(sequences[int(n * 0.9) :], config.model.block_size),
    batch_size=config.training.batch_size,
)

with mlflow.start_run():
    mlflow.log_params(config.data.model_dump())
    mlflow.log_params(config.model.model_dump())
    mlflow.log_params(config.training.model_dump())
    model = NanoGPTSorter(config.data.vocab_size, **config.model.model_dump()).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    for i in range(1, config.training.epoch):
        print(f"Epoch {i + 1}:\n")
        model.train()
        average_t_loss = single_epoch(i, config.training.agg_interval, train_loader)

        model.eval()
        running_v_loss = 0.0
        with torch.no_grad():
            for j, (X, Y) in enumerate(val_loader):
                X = X.to(device)
                Y = Y.to(device)
                pred, loss = model(X, target=Y)
                running_v_loss += loss.item()

        average_v_loss = running_v_loss / len(val_loader)
        mlflow.log_metric("epoch_val_loss", average_v_loss)
        print(f"\nTraining Loss: {average_t_loss}; Validation Loss: {average_v_loss}")

        mlflow_model_info = mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path=f"model_{i + 1}"
        )

    torch.save(model.state_dict(), "weights/model.pt")

    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_loader):
            X = X.to(device)
            Y = Y.to(device)
            pred, loss = model(X, target=Y)
            running_test_loss += loss.item()

    average_test_loss = running_test_loss / len(test_loader)
    print(
        f"\nFinal Training Loss: {average_t_loss}; Validation Loss: {average_v_loss}; Test Loss: {average_test_loss}"
    )
    mlflow.log_metric("test_loss", average_test_loss)
