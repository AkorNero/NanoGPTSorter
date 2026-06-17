import logging
import random

import mlflow
import mlflow.pytorch
import torch
from mlflow.models import infer_signature
from torch.utils.data import DataLoader

from data import vocab
from data.dataset import SorterDataset
from data.tokenizer import encode
from model import config
from model.transformer import NanoGPTSorter


def setup():
    """
    This function sets up mlflow for experiment tracking
    """
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("NanoGPTSorter")
    # mlflow.enable_system_metrics_logging()

    device = (
        "cuda"
        if (config.training.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    example_input = "D B I J G".split(" ")
    example_input = encode(example_input)
    example_input = torch.tensor([example_input], dtype=torch.long, device=device)

    return device, example_input


def single_epoch(
    dataloader: DataLoader,
    isTrain: bool = True,
    agg_interval: int | None = None,
    epoch_index: int | None = None,
):
    total_loss: float = 0.0
    total_token_level_accuracy: float = 0.0
    total_sequence_level_accuracy: float = 0.0

    running_loss: float = 0.0
    running_token_level_accuracy: float = 0.0
    running_sequence_level_accuracy: float = 0.0

    for i, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)

        prediction, loss, tl_accuracy, sl_accuracy = model(X, target=Y)

        loss_val = loss.item()
        total_loss += loss_val
        total_token_level_accuracy += tl_accuracy
        total_sequence_level_accuracy += sl_accuracy

        running_loss += loss_val
        running_token_level_accuracy += tl_accuracy
        running_sequence_level_accuracy += sl_accuracy

        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if agg_interval and (i + 1) % agg_interval == 0:
                interval_loss = running_loss / agg_interval
                interval_tl_acc = running_token_level_accuracy / agg_interval
                interval_sl_acc = running_sequence_level_accuracy / agg_interval

                print(
                    f"\n\tbatch {i + 1} loss: {interval_loss:.2f}; tl accuracy: {interval_tl_acc:.2f}; sl accuracy: {interval_sl_acc:.2f}"
                )

                running_loss = 0.0
                running_token_level_accuracy = 0.0
                running_sequence_level_accuracy = 0.0

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_token_level_accuracy / num_batches,
        total_sequence_level_accuracy / num_batches,
    )


def gen_sequence_dataset(
    num: int, min_seq: int, max_seq: int, train_ratio: float, validation_ratio: float
) -> tuple[
    tuple[SorterDataset, SorterDataset, SorterDataset],
    tuple[DataLoader, DataLoader, DataLoader],
]:
    sequences = set()
    while len(sequences) < num:
        seq_length = random.randint(min_seq, max_seq)
        seq = tuple(
            random.choices(vocab, k=seq_length)
        )  # can not add unhasable list to a set
        sequences.add(seq)

    sequences = [list(s) for s in sequences]
    random.shuffle(sequences)

    n = len(sequences)
    train_dataset = SorterDataset(
        sequences[: int(n * train_ratio)], config.model.block_size
    )
    val_dataset = SorterDataset(
        sequences[int(n * train_ratio) : int(n * (train_ratio + validation_ratio))],
        config.model.block_size,
    )
    test_dataset = SorterDataset(
        sequences[int(n * (train_ratio + validation_ratio)) :], config.model.block_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
    )

    return (
        (train_dataset, val_dataset, test_dataset),
        (train_loader, val_loader, test_loader),
    )


if __name__ == "__main__":
    device, example_input = setup()

    dataset, data_loaders = gen_sequence_dataset(
        config.training.dataset_length,
        config.data.min_seq,
        config.data.max_seq,
        config.training.train_ratio,
        config.training.val_ratio,
    )

    train_ds, validation_ds, test_ds = dataset
    train_loader, val_loader, test_loader = data_loaders

    with mlflow.start_run() as run:
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
            ) = single_epoch(train_loader, True, config.training.agg_interval, i)

            mlflow_model_info = mlflow.pytorch.log_model(
                pytorch_model=model, name=f"model_{i + 1}", step=i + 1
            )

            mlflow.set_active_model(
                name=f"model_{i + 1}",
            )

            mlflow.log_metric(
                "Train/average_loss_per_epoch",
                average_training_loss,
                step=i + 1,
            )
            mlflow.log_metric(
                "Train/average_token_level_accuracy_per_epoch",
                average_token_level_accuracy,
                step=i + 1,
            )
            mlflow.log_metric(
                "Train/average_sequence_level_accuracy_per_epoch",
                average_sequence_level_accuracy,
                step=i + 1,
            )

            model.eval()
            with torch.no_grad():
                (
                    average_v_loss,
                    average_token_level_v_accuracy,
                    average_sequence_level_v_accuracy,
                ) = single_epoch(val_loader, False)

                mlflow.log_metric(
                    "Validation/average_loss_per_epoch",
                    average_v_loss,
                    step=i + 1,
                )
                mlflow.log_metric(
                    "Validation/average_token_level_accuracy_per_epoch",
                    average_token_level_v_accuracy,
                    step=i + 1,
                )
                mlflow.log_metric(
                    "Validation/average_sequence_level_accuracy_per_epoch",
                    average_sequence_level_v_accuracy,
                    step=i + 1,
                )
                print(
                    f"\nTraining Loss: {average_training_loss:.2f}; Validation Loss: {average_v_loss:.2f}; TL Accuracy: {average_token_level_v_accuracy:.2f}; SL Accuracy: {average_sequence_level_v_accuracy:.2f}"
                )

        model_rank = mlflow.search_logged_models(
            order_by=[
                {
                    "field_name": "metrics.Validation/average_sequence_level_accuracy_per_epoch",
                    "ascending": False,
                }
            ],
            output_format="list",
        )

        best_checkpoint = model_rank[0]
        model = mlflow.pytorch.load_model(best_checkpoint.model_uri)

        pred, _, _, _ = model(example_input)
        signature = infer_signature(
            example_input.cpu().numpy(), pred.detach().cpu().numpy()
        )

        mlflow_best_model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=f"best_model_chkpt_{best_checkpoint.name}",
            signature=signature,
            input_example=example_input,
        )

        mlflow.set_active_model(
            name=f"best_model_chkpt_{best_checkpoint.name}",
        )

        torch.save(model.state_dict(), "weights/model.pt")

        model.eval()
        with torch.no_grad():
            (
                average_test_loss,
                average_token_level_test_accuracy,
                average_sequence_level_test_accuracy,
            ) = single_epoch(test_loader, False)

            mlflow.log_metric("Test/average_loss", average_test_loss)
            mlflow.log_metric(
                "Test/average_token_level_accuracy",
                average_token_level_test_accuracy,
            )
            mlflow.log_metric(
                "Test/average_sequence_level_accuracy",
                average_sequence_level_test_accuracy,
            )
            print(
                f"\nFinal Training Loss: {average_test_loss:.2f}; Validation Loss: {average_v_loss:.2f}; Test Loss: {average_test_loss:.2f}; TL Accuracy(test): {average_token_level_test_accuracy:.2f}; SL Accuracy(test): {average_sequence_level_test_accuracy:.2f}"
            )
