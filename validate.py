import torch
import numpy as np

def calculate_iou(predictions, targets):
    """
    Calculate the Intersection over Union (IoU) metric for semantic segmentation.

    Args:
        predictions (torch.Tensor): Predicted segmentation masks.
        targets (torch.Tensor): Ground truth segmentation masks.

    Returns:
        float: IoU score.
    """
    intersection = torch.logical_and(predictions, targets).sum()
    union = torch.logical_or(predictions, targets).sum()
    iou = float(intersection) / float(union)
    return iou

def validate_model(model, validation_loader):
    """
    Validate the model on a validation dataset using IoU.

    Args:
        model (nn.Module): The trained model.
        validation_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        float: Average IoU score for the entire validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    iou_scores = []

    with torch.no_grad():
        for batch in validation_loader:
            images, masks = batch['image'], batch['mask']
            predictions = model(images)

            # Assuming predictions and masks are of shape (batch_size, num_classes, height, width)
            predictions = torch.argmax(predictions, dim=1)  # Convert to class indices
            predictions = (predictions > 0).long()  # Binary prediction masks

            for i in range(len(predictions)):
                iou = calculate_iou(predictions[i], masks[i])
                iou_scores.append(iou)

    average_iou = np.mean(iou_scores)
    return average_iou
