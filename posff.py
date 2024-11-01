import torch
import torch.nn as nn
from torch import Tensor

class PositionWiseFeedForward(nn.Module):
    """
    The class is a subclass of PyTorch's nn.Module, which means it will inherit all functionalities required to work with neural network layers.
    It defines a position-wise feed-forward neural network that consists of two linear layers with a ReLU activation function in between.
    In the context of transformer models, this feed-forward network is applied to each position separately and identically.
    It helps in transforming the features learned by the attention mechanisms within the transformer, acting as an additional processing step for the attention outputs.
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        Initialize the PositionWiseFeedForward layer.
        
        :param d_model: Dimensionality of the model's input and output.
        :param d_ff: Dimensionality of the inner layer in the feed-forward network.
        
        self.fc1 and self.fc2: Two fully connected (linear) layers with input and output dimensions as defined by d_model and d_ff.
        self.relu: ReLU (Rectified Linear Unit) activation function, which introduces non-linearity between the two linear layers.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the PositionWiseFeedForward layer.
        
        :param x: The input to the feed-forward network.
        
        - self.fc1(x): The input is first passed through the first linear layer (fc1).
        - self.relu(...): The output of fc1 is then passed through a ReLU activation function. ReLU replaces all negative values with zeros, introducing non-linearity into the model.
        - self.fc2(...): The activated output is then passed through the second linear layer (fc2), producing the final output.
        
        :return: Output tensor.
        """
        return self.fc2(self.relu(self.fc1(x)))
