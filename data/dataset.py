import random

import torch
from torch.utils.data import Dataset

from . import config, vocab
from .tokenizer import encode, get_eos_id, get_sep_id


class SorterDataset(Dataset):
    def __init__(self, num_samples, block_size):
        super().__init__()
        self.num_samples = num_samples
        self.block_size = block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seq_length = random.randint(1, 5)
        gen_seq = random.choices(vocab, k=seq_length)
        sorted_gen_seq = sorted(gen_seq)
        complete_seq = gen_seq + ["[SEP]"] + sorted_gen_seq + ["[EOS]"]

        no_of_pads = max(0, config.data.block_size - len(complete_seq) + 1)
        complete_seq = complete_seq + ["[PAD]"] * no_of_pads

        encoded_seq = encode(complete_seq)

        x = encoded_seq[:-1]
        y = encoded_seq[1:]

        mask = True

        for i in range(len(y)):
            temp = y[i]
            if mask:
                y[i] = -100
            if (temp == get_sep_id()) or (temp == get_eos_id()):
                mask = not (mask)

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
