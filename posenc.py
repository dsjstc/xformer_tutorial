import torch
import torch.nn as nn
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    """
    The class is defined as a subclass of PyTorch's nn.Module, allowing it to be used as a standard PyTorch layer.
    Adds information about the position of tokens within the sequence. This class helps the model to consider the position of tokens in the sequence.
    The sinusoidal functions used are chosen to allow the model to easily learn to attend to relative positions, as they produce a unique and smooth encoding for each position in the sequence.
    """
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        """
        Initialize the PositionalEncoding layer.
        
        :param d_model: The dimension of the model's input.
        :param max_seq_length: The maximum length of the sequence for which positional encodings are pre-computed.
        
        pe: A tensor filled with zeros, which will be populated with positional encodings.
        position: A tensor containing the position indices for each position in the sequence.
        div_term: A term used to scale the position indices in a specific way.
        The sine function is applied to the even indices and the cosine function to the odd indices of pe.
        Finally, pe is registered as a buffer, which means it will be part of the module's state but will not be considered a trainable parameter.
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the PositionalEncoding layer.
        
        :param x: The input tensor.
        
        The forward method simply adds the positional encodings to the input x.
        It uses the first x.size(1) elements of pe to ensure that the positional encodings match the actual sequence length of x.
        
        :return: Output tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]
