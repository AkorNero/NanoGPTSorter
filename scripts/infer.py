import sys

import torch

from data.tokenizer import decode, encode, get_eos_id, get_sep_id
from model import config
from model.transformer import NanoGPTSorter

device = (
    "cuda"
    if (config.training.device == "cuda" and torch.cuda.is_available())
    else "cpu"
)

infer_trained_model = NanoGPTSorter(config.data.vocab_size, **config.model.model_dump())
infer_trained_model.load_state_dict(torch.load("weights/model.pt", map_location=device))
infer_trained_model.to(device)
infer_trained_model.eval()

sequence = None

if sys.argv[1:]:
    sequence = sys.argv[1:]
else:
    sequence = input(
        "Enter a sequence of characters to sort (A-J, length 1-5, seperated by space): "
    ).split(" ")
if (1 <= len(sequence) and len(sequence) <= 5) and all(
    (ord("A") <= ord(i) and ord(i) <= ord("J")) for i in sequence
):
    print(f"Original Sequence: {' '.join(sequence)}")
    sequence.append("[SEP]")
    enc_seq = encode(sequence)
    enc_seq_tensor = torch.tensor(
        data=[enc_seq], dtype=torch.long, device=device
    )  # (1,seq_len)

    with torch.no_grad():
        while True:
            logits, _, _, _ = infer_trained_model(enc_seq_tensor)
            next_token_logits = logits[
                0, -1, :
            ]  # (vocab_size,) last token vocab prediction logits

            next_token_id = torch.argmax(
                next_token_logits, dim=-1
            )  # gives index of the highly probable token

            if (
                next_token_id == get_eos_id()
                or enc_seq_tensor.size(1) >= config.model.block_size
            ):
                break

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            enc_seq_tensor = torch.cat((enc_seq_tensor, next_token_tensor), dim=1)

        output_list = enc_seq_tensor[0].tolist()
        idx = output_list.index(get_sep_id())
        dec_seq = decode(output_list[idx + 1 :])

        print(f"Sorted Sequence: {' '.join(dec_seq)}")
