import torch
import torch.nn as nn
from mha import MultiHeadAttention
from posff import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head self-attention mechanism followed by position-wise feed-forward neural network, with residual connections, layer normalization, and dropout applied as appropriate. These components together allow the encoder to capture complex relationships in the input data and transform them into a useful representation for downstream tasks. Typically, multiple such encoder layers are stacked to form the complete encoder part of a transformer model.

    Parameters:
    - d_model (int): The dimensionality of the input.
    - num_heads (int): The number of attention heads in the multi-head attention.
    - d_ff (int): The dimensionality of the inner layer in the position-wise feed-forward network.
    - dropout (float): The dropout rate used for regularization.

    Components:
    - self.self_attn: Multi-head attention mechanism.
    - self.feed_forward: Position-wise feed-forward neural network.
    - self.norm1 and self.norm2: Layer normalization.
    - self.dropout: Dropout layer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward method for the encoder layer.

        Parameters:
        - x (torch.Tensor): The input to the encoder layer.
        - mask (torch.Tensor, optional): Optional mask to ignore certain parts of the input.

        Processing Steps:
        1. Self-Attention: The input x is passed through the multi-head self-attention mechanism.
        2. Add & Normalize (after Attention): The attention output is added to the original input (residual connection), followed by dropout and normalization.
        3. Feed-Forward Network: The output from the previous step is passed through the position-wise feed-forward network.
        4. Add & Normalize (after Feed-Forward): The feed-forward output is added to the input of this stage (residual connection), followed by dropout and normalization.

        Returns:
        - torch.Tensor: The processed tensor as the output of the encoder layer.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
