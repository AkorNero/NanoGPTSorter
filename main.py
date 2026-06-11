import data
from data import dataset, tokenizer


def main():
    # print(data.config)
    # test_seq = ["B", "C", "A", "[SEP]", "A", "B", "C", "[EOS]"]
    # print(f"Test Sequence: {test_seq}")

    # encoded_seq = tokenizer.encode(test_seq)
    # print(encoded_seq)

    # decoded_seq = tokenizer.decode(encoded_seq)
    # print(decoded_seq)

    # assert test_seq == decoded_seq, "Decoding failed to match original sequence"

    dataset_gen = dataset.SorterDataset(100, 12)
    dataset_gen.__getitem__(3)


if __name__ == "__main__":
    main()
