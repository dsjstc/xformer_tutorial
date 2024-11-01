# %%
import glob
import datasets
from datasets import Dataset, DatasetDict
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from trans import SemanticSegmentationTransformer

if torch.cuda.is_available():
    # Set the default device to CUDA
    torch.cuda.set_device(0)  # You can specify the GPU index if you have multiple GPUs
    print("Using CUDA")
else:
    print("CUDA is not available")

# monkeypatch to show tensor shapes in repr
def custom_repr(self):
    return f'{{T{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


# Settings
batch_size = 1
learning_rate = 0.1
num_epochs = 10
validation_interval = 1

# %%
def get_data(split, data_dir):
    img_paths = glob.glob(f"{data_dir}/Images/img_dir/{split}/*.jpg")
    ann_paths = [p.replace("img_dir", "ann_dir").replace(".jpg", ".png") for p in img_paths]
    return {"image_path": img_paths, "annotation_path": ann_paths}

data_dir = Path("../datasets/FoodSeg103")

train_dict = get_data("train", data_dir)
test_dict = get_data("test", data_dir)
train_splitme = Dataset.from_dict(train_dict)
train_dict, validation_dict = train_test_split(train_splitme, test_size=0.2, random_state=42)

dataset = DatasetDict({
    "train": Dataset.from_dict(train_dict),
    "validation": Dataset.from_dict(validation_dict),
    "test": Dataset.from_dict(test_dict)
})

# %%
from imageloader import ImageLoader
train_loader = ImageLoader(dataset['train'], batch_size=batch_size, shuffle=True)
testid_loader = ImageLoader(dataset['validation'], batch_size=batch_size, shuffle=False)
test_loader = ImageLoader(dataset['test'], batch_size=batch_size, shuffle=False)

# %%
device='cuda'
model = SemanticSegmentationTransformer(num_classes=104, input_size=3*512*512, d_model=1024, num_heads=8, 
                          num_encoder_layers=24, num_decoder_layers=24, d_ff=4096, max_seq_length=512^2, 
                          dropout=0.1)
model.to(device)



# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_validation_loss = float('inf')

# %%
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    # Use tqdm for a progress bar during training
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Ensure inputs and targets are on the same device as the model
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Create source and target masks
        src_mask = torch.ones_like(inputs)  # You need to implement create_source_mask
        tgt_mask = torch.ones_like(targets)  # You need to implement create_target_mask
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with masks
        outputs = model(inputs, targets, src_mask, tgt_mask)  # Adjust your forward pass
        
        # Compute the loss
        loss = criterion(outputs, targets)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")

    # Validation and model checkpointing (optional)
    if (epoch + 1) % validation_interval == 0:
        model.eval()
        with torch.no_grad():
            total_validation_loss = 0.0
            for val_inputs, val_targets in valid_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                # Create source and target masks for validation
                val_src_mask = torch.ones_like(val_inputs)  
                val_tgt_mask = torch.ones_like(val_targets)  

                
                val_outputs = model(val_inputs, val_targets, val_src_mask, val_tgt_mask)  # Adjust forward pass
                val_loss = criterion(val_outputs, val_targets)
                total_validation_loss += val_loss.item()
        
        # Calculate average validation loss
        average_val_loss = total_validation_loss / len(valid_loader)
        print(f"Validation Loss: {average_val_loss:.4f}")

        # Check if validation performance improves and save the model if needed
        if average_val_loss < best_validation_loss:
            best_validation_loss = average_val_loss
            torch.save(model.state_dict(), 'best_model.pt')  # Save the best model



# %%
# Testing the trained model (on a separate test dataset)
model.load_state_dict(torch.load('best_model.pt'))  # Load the best model
model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        
        # Create source and target masks for testing
        val_src_mask = torch.ones_like(val_inputs)  
        val_tgt_mask = torch.ones_like(val_targets)  

        
        test_outputs = model(test_inputs, test_targets, test_src_mask, test_tgt_mask)  # Adjust forward pass
        test_loss += criterion(test_outputs, test_targets).item()

        _, predicted = test_outputs.max(1)
        total += test_targets.size(0)
        correct += predicted.eq(test_targets).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# %%


# %%



