import logging
import random

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader

from data import vocab
from data.dataset import SorterDataset
from model import config
from model.transformer import NanoGPTSorter


def setup():
    """
    This function sets up mlflow for experiment tracking
    """
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("NanoGPTSorter")
    mlflow.enable_system_metrics_logging()


def single_epoch(epoch_index: int, agg_interval: int, dataloader: DataLoader):
    running_loss: float = 0.0
    average_loss: float = 0.0
    running_token_level_accuracy = 0.0
    average_token_level_accuracy = 0.0
    running_sequence_level_accuracy = 0.0
    average_sequence_level_accuracy = 0.0
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(
            device
        )  # unlike nn.Module methods the tesor.to() method returns a new tensor
        Y = Y.to(device)
        optimizer.zero_grad()
        prediction, loss, tl_accuracy, sl_accuracy = model(X, target=Y)
        loss.backward()
        optimizer.step()

        log_idx = epoch_index * len(dataloader) + i + 1

        running_loss += loss.item()
        running_token_level_accuracy += tl_accuracy
        running_sequence_level_accuracy += sl_accuracy
        if (i + 1) % agg_interval == 0:
            average_loss = running_loss / agg_interval
            average_token_level_accuracy = running_token_level_accuracy / agg_interval
            average_sequence_level_accuracy = (
                running_sequence_level_accuracy / agg_interval
            )

            print(
                f"\n\tbatch {i + 1} loss: {average_loss:.2f}; tl accuracy: {average_token_level_accuracy:.2f}; sl accuracy: {average_sequence_level_accuracy:.2f}"
            )
            mlflow.log_metric(
                f"average_training_loss_per_{agg_interval}_batch",
                average_loss,
                step=log_idx,
            )
            mlflow.log_metric(
                f"average_token_level_training_accuracy_per_{agg_interval}_batch",
                average_token_level_accuracy,
                step=log_idx,
            )
            mlflow.log_metric(
                f"average_sequence_level_training_accuracy_per_{agg_interval}_batch",
                average_sequence_level_accuracy,
                step=log_idx,
            )

            running_loss = 0.0
            running_token_level_accuracy = 0.0
            running_sequence_level_accuracy = 0.0
    return average_loss, average_token_level_accuracy, average_sequence_level_accuracy


if __name__ == "__main__":
    device = (
        "cuda"
        if (config.training.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    setup()

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
    train_dataset = sequences[: int(n * 0.8)]
    val_dataset = sequences[int(n * 0.8) : int(n * 0.9)]
    test_dataset = sequences[int(n * 0.9) :]

    train_loader = DataLoader(
        SorterDataset(train_dataset, config.model.block_size),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SorterDataset(val_dataset, config.model.block_size),
        batch_size=config.training.batch_size,
    )
    test_loader = DataLoader(
        SorterDataset(test_dataset, config.model.block_size),
        batch_size=config.training.batch_size,
    )

    with mlflow.start_run():
        mlflow.log_params(config.data.model_dump())
        mlflow.log_params(config.model.model_dump())
        mlflow.log_params(config.training.model_dump())
        model = NanoGPTSorter(config.data.vocab_size, **config.model.model_dump()).to(
            device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.training.learning_rate
        )

        for i in range(0, config.training.epoch):
            print(f"Epoch {i + 1}:\n")
            model.train()
            (
                average_training_loss,
                average_token_level_accuracy,
                average_sequence_level_accuracy,
            ) = single_epoch(i, config.training.agg_interval, train_loader)

            mlflow.log_metric(
                "average_training_loss_per_epoch", average_training_loss, step=i + 1
            )
            mlflow.log_metric(
                "average_token_level_training_accuracy_per_epoch",
                average_token_level_accuracy,
                step=i + 1,
            )
            mlflow.log_metric(
                "average_sequence_level_training_accuracy_per_epoch",
                average_sequence_level_accuracy,
                step=i + 1,
            )

            model.eval()
            running_v_loss = 0.0
            running_token_level_v_accuracy = 0.0
            running_sequence_level_v_accuracy = 0.0
            with torch.no_grad():
                for j, (X, Y) in enumerate(val_loader):
                    X = X.to(device)
                    Y = Y.to(device)
                    pred, loss, tl_acc, sl_acc = model(X, target=Y)
                    running_v_loss += loss.item()
                    running_token_level_v_accuracy += tl_acc
                    running_sequence_level_v_accuracy += sl_acc

            average_v_loss = running_v_loss / len(val_loader)
            average_token_level_v_accuracy = running_token_level_v_accuracy / len(
                val_loader
            )
            average_sequence_level_v_accuracy = running_sequence_level_v_accuracy / len(
                val_loader
            )
            mlflow.log_metric(
                "average_validation_loss_per_epoch", average_v_loss, step=i + 1
            )
            mlflow.log_metric(
                "average_token_level_valiation_accuracy_per_epoch",
                average_token_level_v_accuracy,
                step=i + 1,
            )
            mlflow.log_metric(
                "average_sequence_level_validation_accuracy_per_epoch",
                average_sequence_level_v_accuracy,
                step=i + 1,
            )
            print(
                f"\nTraining Loss: {average_training_loss:.2f}; Validation Loss: {average_v_loss:.2f}; TL Accuracy: {average_token_level_v_accuracy:.2f}; SL Accuracy: {average_sequence_level_v_accuracy:.2f}"
            )

            mlflow_model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"model_{i + 1}",
            )

        torch.save(model.state_dict(), "weights/model.pt")

        model.eval()
        running_test_loss = 0.0
        running_token_level_test_accuracy = 0.0
        running_sequence_level_test_accuracy = 0.0
        with torch.no_grad():
            for i, (X, Y) in enumerate(test_loader):
                X = X.to(device)
                Y = Y.to(device)
                pred, loss, tl_acc, sl_acc = model(X, target=Y)
                running_test_loss += loss.item()
                running_token_level_test_accuracy += tl_acc
                running_sequence_level_test_accuracy += sl_acc

        average_test_loss = running_test_loss / len(test_loader)
        average_token_level_test_accuracy = running_token_level_test_accuracy / len(
            test_loader
        )
        average_sequence_level_test_accuracy = (
            running_sequence_level_test_accuracy / len(test_loader)
        )
        mlflow.log_metric("average_test_loss", average_test_loss)
        mlflow.log_metric(
            "average_token_level_test_accuracy", average_token_level_test_accuracy
        )
        mlflow.log_metric(
            "average_sequence_level_test_accuracy", average_sequence_level_test_accuracy
        )
        print(
            f"\nFinal Training Loss: {average_test_loss:.2f}; Validation Loss: {average_v_loss:.2f}; Test Loss: {average_test_loss:.2f}; TL Accuracy(test): {average_token_level_test_accuracy:.2f}; SL Accuracy(test): {average_sequence_level_test_accuracy:.2f}"
        )
