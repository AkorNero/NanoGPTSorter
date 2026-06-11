import os
import string
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401

import utils  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
CONFIG_PATH = os.environ.get("CONFIG_PATH")

config = utils.conv_yaml_dc(CONFIG_PATH)

vocab = list(
    string.ascii_uppercase[: config.data.vocab_size - config.data.special_characters]
)

stoi = {v: i + 1 for i, v in enumerate(vocab)}
stoi["[PAD]"] = 0
stoi["[SEP]"] = len(vocab) + 1
stoi["[EOS]"] = len(vocab) + 2
itos = {v: k for k, v in stoi.items()}
