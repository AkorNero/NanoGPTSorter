from . import itos, stoi


def encode(input) -> list[int]:
    return [stoi[char] for char in input]


def decode(input: list[int]) -> list[str]:
    return [itos[idx] for idx in input]


def get_pad_id():
    return stoi["[PAD]"]


def get_sep_id():
    return stoi["[SEP]"]


def get_eos_id():
    return stoi["[EOS]"]
