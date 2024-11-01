from datasets import DatasetDict, Dataset
import os
import glob

def load_dataset(data_dir):
    def foodseg103_loader(split):
        images = []
        annotations = []
        labels = []
        
        img_dir = os.path.join(data_dir, "Images", "img_dir", split)
        ann_dir = os.path.join(data_dir, "Images", "ann_dir", split)
        
        for class_dir in glob.glob(f"{img_dir}/*"):
            class_name = os.path.basename(class_dir)
            for img_path in glob.glob(f"{class_dir}/*.jpg"):
                ann_path = os.path.join(ann_dir, class_name, os.path.basename(img_path).replace(".jpg", ".png"))
                images.append(img_path)
                annotations.append(ann_path)
                labels.append(class_name)
                
        return {
            "image_path": images,
            "annotation_path": annotations,
            "label": labels
        }

    train_data = foodseg103_loader("train")
    test_data = foodseg103_loader("test")
    
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })


