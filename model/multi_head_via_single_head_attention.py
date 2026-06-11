import torch

# from . import config


class SingleHeadCausalSelfAttention(torch.nn.Module):
    def __init__(self, block_size, d_model, a_model):
        super().__init__()
        self.t_model = block_size
        self.d_model = d_model
        self.a_model = a_model

        """
            Pytorch Applies an affine linear transformation to the incoming data: y=xA^T+b.
            Therefore the input dimension should be of form (batch x token x dimension) for this to work
            Simply put the linear transformation is done from last dimension of the input to the first dimension of Weight
        """
        # Q, K and V weight and bias
        self.Qvecs = torch.nn.Linear(self.d_model, self.a_model)
        self.Kvecs = torch.nn.Linear(self.d_model, self.a_model)
        self.Vvecs = torch.nn.Linear(self.d_model, self.a_model)

        lower_triangular_mask = torch.tril(torch.ones(self.t_model, self.t_model))

        self.register_buffer("lower_triangular_mask", lower_triangular_mask)

    def forward(self, x):
        input_sequence_length = x.size(1)
        Q = self.Qvecs(x)
        K = self.Kvecs(x)
        V = self.Vvecs(x)

        """
            we calculate the dot products for all time steps simultaneously using matrix multiplication, and then we force the "past-only" rule by applying a lower triangular mask.
        """

        attention_matrix = torch.matmul(Q, K.transpose(-2, -1))

        # print(lower_triangular_mask == 0)

        masked_attention_matrix = attention_matrix.masked_fill(
            self.lower_triangular_mask[:input_sequence_length, :input_sequence_length]  # ty:ignore[not-subscriptable]
            == 0,
            float("-inf"),
        )

        softmax_attention_matrix = torch.softmax(
            masked_attention_matrix / (self.a_model**0.5), dim=-1
        )

        output = softmax_attention_matrix @ V

        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, block_size, d_model, n_heads):
        super().__init__()
        self.t_model = block_size
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.a_model = d_model // n_heads

        self.multihead = torch.nn.ModuleList(
            [
                SingleHeadCausalSelfAttention(self.t_model, self.d_model, self.a_model)
                for i in range(n_heads)
            ]
        )

        self.projection = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        output = []
        for l in self.multihead:  # noqa: E741
            output.append(l(x))

        stacked_v_output = torch.cat(output, -1)

        return self.projection(stacked_v_output)
