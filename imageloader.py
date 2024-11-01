import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ImageLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, image_size=(512, 512), normalize=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.normalize = normalize
        self.data_loader = self.create_data_loader()

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        image = transforms.Resize(self.image_size)(image)
        image = transforms.ToTensor()(image)
        if self.normalize:
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image

    def preprocess_mask(self, mask_path):
        # Load and preprocess the segmentation mask
        mask = Image.open(mask_path)
        mask = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(mask)
        mask = np.array(mask)
        return torch.tensor(mask, dtype=torch.long)

    def tokenize_image_pair(self, image_path, mask_path):
        # Preprocess image and mask
        image = self.preprocess_image(image_path)
        mask = self.preprocess_mask(mask_path)
        return image, mask

    def create_data_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,  # You can adjust the number of workers for data loading
            collate_fn=self.collate_fn  # Custom collate function for preprocessing
        )

    def collate_fn(self, batch):
        images, masks = zip(*[(self.preprocess_image(sample['image_path']), self.preprocess_mask(sample['annotation_path'])) for sample in batch])
        images = torch.stack(images)
        masks = torch.stack(masks)
        return {'image': images, 'mask': masks}

    def __iter__(self):
        for batch in self.data_loader:
            yield batch['image'], batch['mask']

    def __len__(self):
        return len(self.data_loader)
