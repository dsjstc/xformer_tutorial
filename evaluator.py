import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score  # Use appropriate metrics for segmentation

class SegmentationEvaluator:
    def __init__(self, tokenizer, transformer_model):
        self.tokenizer = tokenizer
        self.transformer_model = transformer_model

    def evaluate(self, dataset, batch_size=32):
        # Tokenize the dataset using the provided tokenizer
        tokenized_dataset = self.tokenizer.tokenize_dataset(dataset)

        # Create a DataLoader for batching
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        # Initialize lists to store predictions and ground truth labels
        predictions = []
        labels = []

        # Set the model to evaluation mode
        self.transformer_model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image']
                masks = batch['mask']

                # Forward pass through the transformer model
                predicted_masks = self.transformer_model(images)

                # Append predicted masks and ground truth labels
                predictions.extend(predicted_masks)
                labels.extend(masks)

        # Convert predictions and labels to numpy arrays
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        # Calculate evaluation metrics (e.g., accuracy for semantic segmentation)
        # You can replace accuracy_score with appropriate segmentation metrics
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())

        return accuracy
