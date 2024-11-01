import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import binary_erosion, label
from scipy.ndimage import distance_transform_edt
from pathlib import Path

class MaskOverlayDisplay:
    def __init__(self, dataset, category_id_path=None, rows=1, cols=2, border_width=7, transparency=0.5):
        self.dataset = dataset
        self.rows = rows if rows else 1
        self.cols = cols if cols else 2
        self.border_width = border_width if border_width else 7
        self.transparency = transparency if transparency else 0.5
        self.category_id_path = category_id_path if category_id_path else "../datasets/FoodSeg103/category_id.txt"
        self.class_labels = self.read_category_ids()

        # Define a custom colormap with transparency for value 0
        cmap = plt.cm.get_cmap('tab10', len(self.class_labels))
        new_colors = [list(cmap(i)) for i in range(len(self.class_labels))]
        new_colors[0] = [0, 0, 0, 0]  # Set transparency (alpha) to 0 for value 0
        self.color_map = plt.cm.colors.ListedColormap(new_colors, name='custom_cmap')

        # Initialize member variables for images and masks
        self.images = []
        self.masks = []
        self.hollowed_masks = []

    def read_category_ids(self):
        with open(self.category_id_path, 'r') as f:
            return {int(line.split('\t')[0]): line.split('\t')[1].strip() for line in f.readlines()}

    def hollow_out_regions(self, mask, border_width):
        # Label connected components in the mask
        labeled_mask, num_features = label(mask, return_num=True)

        # Initialize an empty hollowed mask
        hollowed_mask = np.zeros_like(mask, dtype=np.uint8)  # Ensure the data type is explicitly set to uint8

        # Iterate through each connected component
        for label_value in range(1, num_features + 1):
            component_mask = (labeled_mask == label_value).astype(np.uint8)

            # Calculate the distance transform of the component mask
            distance_transform = distance_transform_edt(component_mask)

            # Create a mask by eroding the distance transform with a circular structuring element
            border_mask = distance_transform > border_width

            # Add the border_mask to the hollowed mask while preserving the data type
            hollowed_mask = np.maximum(hollowed_mask, (border_mask * label_value).astype(np.uint8))

        return hollowed_mask

    def load_samples(self, sample_ids):
        # Clear existing data
        self.images.clear()
        self.masks.clear()
        self.hollowed_masks.clear()

        for sample_id in sample_ids:
            img_path = Path(self.dataset[int(sample_id)]['image_path'])
            ann_path = Path(self.dataset[int(sample_id)]['annotation_path'])
            img = cv2.imread(str(img_path))
            ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
            self.images.append(img)
            self.masks.append(ann)
            self.hollowed_masks.append(self.hollow_out_regions(ann, self.border_width))

    def display_samples(self):
        fig, axes = plt.subplots(self.rows, 2, figsize=(15, 8))
        if self.rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for i in range(len(self.images)):
            overlay = self.images[i].copy()
            hollowed_mask = self.hollowed_masks[i]
            for label in np.unique(hollowed_mask):
                if label == 0:
                    continue
                overlay[hollowed_mask == label] = (np.array(self.color_map(label))[:3] * 255).astype(np.uint8)

            ax1, ax2 = axes[i // self.cols]
            ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax2.imshow(hollowed_mask, cmap=self.color_map)
            unique_labels = np.unique(hollowed_mask)
            legend_labels = [self.class_labels[label] for label in unique_labels if label in self.class_labels]
            ax2.legend(legend_labels)
            ax1.set_axis_off()
            ax2.set_axis_off()
        plt.show()

    def display_random_samples(self, num_samples=2):
        random_ids = np.random.choice(len(self.dataset), num_samples, replace=False)
        print(f"Displaying samples for IDs: {random_ids}")
        self.load_samples(random_ids)
        self.display_samples()
