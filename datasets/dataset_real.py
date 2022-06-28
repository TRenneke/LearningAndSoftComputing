import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class2id = {"stop": 1, "yield": 2, "yieldAhead": 3, "merge": 4, "signalAhead": 5, "pedestrianCrossing": 6, "keepRight": 7, "speedLimit35": 8, "speedLimit25":9}
# dataset definition
class myDataset(Dataset):
    # load the dataset
    def __init__(self,root,transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        #with open(os.path.join(self.root, "annotations.csv")) as file:
        #    self.data = csv.reader(self.data)
        self.data = pd.read_csv(os.path.join(self.root, "annotations.csv"))
    # get a row at an index
    def __getitem__(self, idx):
        idx = idx -1
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        row = self.data.iloc[idx]
        
        boxes = []
        
        x1 = int(row[1])
        y1 = int(row[2])
        x2 = int(row[3])
        y2 = int(row[4])
        label = class2id[row[5]]#int(row[5])
        boxes.append([x1, y1, x2, y2])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.tensor([False])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)