import torch

from .multi_head_via_single_head_attention import MultiHeadAttention


class MLP(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, 4 * d_model)
        self.activation = torch.nn.GELU()

        self.linear_2 = torch.nn.Linear(4 * d_model, d_model)  # projection

    def forward(self, x):
        return self.linear_2(self.activation(self.linear_1(x)))


class TransformerBlock(torch.nn.Module):
    def __init__(self, block_size, d_model, n_heads):
        super().__init__()
        self.layer_norm_attention = torch.nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(block_size, d_model, n_heads)

        self.layer_norm_mlp = torch.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm_attention(x))
        x = x + self.mlp(self.layer_norm_mlp(x))
        return x


class NanoGPTSorter(torch.nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_embedding = torch.nn.Embedding(block_size, d_model)
        self.transformer_block = torch.nn.ModuleList(
            [TransformerBlock(block_size, d_model, n_heads) for _ in range(n_layers)]
        )
        self.layer_norm_final = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, target=None):
        # B = idx.size(0)
        T = idx.size(1)
        token_embedding = self.token_embedding(idx)  # (B,T,d_model)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_embedding = self.positional_embedding(pos)  # (T, d_model)

        x = token_embedding + pos_embedding  # pos_embedding is broadcasted

        for block in self.transformer_block:
            x = block(x)

        x = self.layer_norm_final(x)  # (B, T, d_model)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if target is not None:
            logits_flat = logits.view(
                -1, logits.size(-1)
            )  # (B*T, vocab_size) here -1 argument of view method specifies to consilidate that particular dimension on its own base on possibility here it B*T
            target_flat = target.view(-1)  # (B*T,)

            loss = torch.nn.functional.cross_entropy(
                logits_flat, target_flat, ignore_index=-100
            )

            # token-level accuracy - The percentage of individual chatacter that the model predicted correctly
            # sequence-level accuracy - The percentage of sequence that were sorted completely correct

            preds = torch.argmax(
                logits, dim=-1
            )  # (B, T) predicted token for the entire sequence or context range
            mask = target != -100  # mask to ignore the input and pad token

            correct_tokens = (preds == target) & mask  # (B, T)
            # here the bitwise & operator acts as an element wise and operator due to pytorch operator overloading

            token_level_accuracy = (
                correct_tokens.sum().item() / mask.sum().item()
            )  # float
            # .item() method extracts the raw value from a tensor containing a single element and converts it into a standard Python int or float.

            is_completely_correct = (preds == target) | (target == -100)  # (B, T)
            # preds should either match with the target (or) must target must be -100 for the element to be true. here also the bitwise | acts as an element wise or operator

            sequence_level_accuracy = (
                is_completely_correct.all(dim=-1).float().mean().item()
            )  # (B,)

            return logits, loss, token_level_accuracy, sequence_level_accuracy
        return logits, None, None, None
