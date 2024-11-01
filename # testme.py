# testme

import torch
import torch.nn as nn

# Set the default device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

# Define and create the linear layer on the GPU
output_linear = nn.Linear(64, 64).to(device)

# Check the device of the linear layer
print(output_linear.weight.device)  # This should print 'cuda:0'
