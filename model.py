import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

HT_CLASS_MAP = {
    0: "jug",       # Crimp?
    1: "crimp",     # Pinch? or jug?
    2: "pinch",     # pocket?
    3: "pocket",    # 
    4: "slope",
    5: "volume"
}

HT_TEST_IMG_PATH = "data/hold-type/All_Data/images"
HT_TEST_ANN_PATH = "data/hold-type/All_Data/labels"

class HoldTypeDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        ann_path = os.path.join(self.ann_dir, self.img_files[index].replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    type_id, x, y, bw, bh = map(float, line.strip().split())
                    
                    x1 = (x - bw / 2) * w
                    y1 = (y - bh / 2) * h
                    x2 = (x + bw / 2) * w
                    y2 = (y + bh / 2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(type_id))
                    
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)   
        
        target = {'boxes': boxes, 'labels': labels}
        
        if self.transform:
            image = self.transform(image)
            
        return image, target
    
def visualize_ht_sample(image, target):
    image = np.array(image).copy()
    boxes = target['boxes'].numpy().astype(int)
    labels = target['labels'].numpy()
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        label = labels[i].item()
        text_label = HT_CLASS_MAP.get(label, "unknown")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text_label, (x1, y1 - 10), 0, 0.9, (0, 255, 0), 2)
        
    cv2.imshow('Hold Type Sample', image)
    cv2.waitKey(0)
    
def collate_fn(batch):
    return tuple(zip(*batch))
           
        
hold_type_dataset = HoldTypeDataset(img_dir=HT_TEST_IMG_PATH, ann_dir=HT_TEST_ANN_PATH)
hold_type_dataloader = DataLoader(hold_type_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

while True:
    for images, targets in hold_type_dataloader:
        for img, tgt in zip(images, targets):
            visualize_ht_sample(img, tgt)