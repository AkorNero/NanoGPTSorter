import string

from dotenv import load_dotenv  # noqa: F401

from model import config

vocab = list(
    string.ascii_uppercase[: config.data.vocab_size - config.data.special_characters]
)

stoi = {v: i + 1 for i, v in enumerate(vocab)}
stoi["[PAD]"] = 0
stoi["[SEP]"] = len(vocab) + 1
stoi["[EOS]"] = len(vocab) + 2
itos = {v: k for k, v in stoi.items()}
