import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    """
    The class is defined as a subclass of PyTorch's nn.Module.
    It encapsulates the multi-head attention mechanism commonly used in transformer models.
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Initialize the MultiHeadAttention layer.
        
        :param d_model: Dimensionality of the input.
        :param num_heads: Number of attention heads to split the input into.
        The initialization checks if d_model is divisible by num_heads, and then defines the transformation weights for query, key, value, and output.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate scaled dot-product attention.
        
        :param Q: Query tensor.
        :param K: Key tensor.
        :param V: Value tensor.
        :param mask: Mask tensor.
        :return: Output tensor.
        
        - Calculating Attention Scores: attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k).
        - Applying Mask: If a mask is provided, it is applied to the attention scores to mask out specific values.
        - Calculating Attention Weights: The attention scores are passed through a softmax function to convert them into probabilities that sum to 1.
        - Calculating Output: The final output of the attention is calculated by multiplying the attention weights by the values (V).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input to have multiple heads for multi-head attention.
        
        :param x: Input tensor.
        :return: Reshaped tensor.
        
        This method reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k).
        It enables the model to process multiple attention heads concurrently, allowing for parallel computation.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input to have a single head for multi-head attention.
        
        :param x: Input tensor.
        :return: Reshaped tensor.
        
        This method reshapes the input x into the shape (batch_size, seq_length, d_model).
        It combines the multiple attention heads into a single tensor.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply multi-head attention to the input tensors.
        
        :param Q: Query tensor.
        :param K: Key tensor.
        :param V: Value tensor.
        :param mask: Mask tensor.
        :return: Output tensor.
        
        This method applies multi-head attention to the input tensors Q, K, and V.
        It first splits the tensors into multiple heads, applies attention to each head, and then combines the heads into a single tensor.
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.W_o(self.combine_heads(attn_output))
        return output