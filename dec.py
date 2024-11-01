import torch
import torch.nn as nn
from mha import MultiHeadAttention
from posff import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Defines a single layer of the transformer's decoder.
    
    Parameters:
    - d_model (int): Dimensionality of the input.
    - num_heads (int): Number of attention heads in multi-head attention.
    - d_ff (int): Dimensionality of inner layer in feed-forward network.
    - dropout (float): Dropout rate for regularization.
    
    Components:
    - self.self_attn: Multi-head self-attention for target sequence.
    - self.cross_attn: Multi-head attention for encoder's output.
    - self.feed_forward: Position-wise feed-forward network.
    - self.norm1, self.norm2, self.norm3: Layer normalization.
    - self.dropout: Dropout layer for regularization.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward method for the decoder layer.
        
        Parameters:
        - x (torch.Tensor): Input to the decoder layer.
        - enc_output (torch.Tensor): Output from corresponding encoder.
        - src_mask (torch.Tensor, optional): Source mask for encoder's output.
        - tgt_mask (torch.Tensor, optional): Target mask for decoder's input.
        
        Processing Steps:
        1. Self-Attention on Target Sequence
        2. Add & Normalize (after Self-Attention)
        3. Cross-Attention with Encoder Output
        4. Add & Normalize (after Cross-Attention)
        5. Feed-Forward Network
        6. Add & Normalize (after Feed-Forward)
        
        Returns:
        - torch.Tensor: Processed tensor as output of decoder layer.
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
