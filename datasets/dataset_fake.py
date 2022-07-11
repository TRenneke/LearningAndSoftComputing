<<<<<<< HEAD
from datetime import MAXYEAR
=======
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
import random 
from torch.utils.data import Dataset
import os
import cv2 as cv
from PIL import Image
import numpy as np
import torch as torch
import json

<<<<<<< HEAD
def bb_union(bb1, bb2):
    minx1, miny1, maxx1, maxy1 = bb1
    minx2, miny2, maxx2, maxy2 = bb2
    minx, miny, maxx, maxy = max(minx1, minx2), max(miny1, miny2), min(maxx1, maxx2), min(maxy1, maxy2)
    if maxx < minx or maxy < miny:
        return 0, 0, 0, 0
    return minx, miny, maxx, maxy
def bb_area(bb):
    minx, miny, maxx, maxy = bb
    return (maxx - minx) * (maxy - miny)
def bb_oclusion(bb1, bb2):
    return bb_area(bb_union(bb1, bb2)) / min(bb_area(bb1), bb_area(bb2))

=======
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
class fake_dataset(Dataset):
    def __init__(self, sign_path: str, background_path: str, length: int, 
                 scr: tuple[int, int], size: int, sign_sice: tuple[int, int], transforms = None):
        self.sign_path = sign_path
        self.background_path = background_path
        self.signs = os.listdir(os.path.join(sign_path, "imgs"))
        self.backgrounds = os.listdir(background_path)
        self.length = length
        self.scr = scr
        self.sign_size = sign_sice
        self.transforms = transforms
        self.name2label = {"Stop": 1, "Yield": 2, "YieldAhead": 3, "Merge": 4, "TrafficLightAhead": 5, "PedestrianCrossing": 6, "KeepRight": 7, "MaxSpeed35": 8, "MaxSpeed25": 9}
        self.edge_points = {1: [np.array([0.4, 1, 0, 1]), np.array([-0.4, 1, 0, 1]), np.array([0.4, -1, 0, 1]), np.array([-0.4, -1, 0, 1]),
                                np.array([1, 0.4, 0, 1]), np.array([1, -0.4, 0, 1]), np.array([-1, 0.4, 0, 1]), np.array([-1, -0.4, 0, 1])],
                            2: [np.array([-1, 1, 0, 1]), np.array([1, 1, 0, 1]), np.array([0, -1, 0, 1])],
                            3: [np.array([1, 0, 0, 1]), np.array([-1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, -1, 0, 1])],
                            4: [np.array([1, 0, 0, 1]), np.array([-1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, -1, 0, 1])],
                            5: [np.array([1, 0, 0, 1]), np.array([-1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, -1, 0, 1])],
                            6: [np.array([1, 0, 0, 1]), np.array([-1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, -1, 0, 1])],
                            7: [np.array([1, 1, 0, 1]), np.array([-1, 1, 0, 1]), np.array([1, -1, 0, 1]), np.array([-1, -1, 0, 1])],
                            8: [np.array([1, 1, 0, 1]), np.array([-1, 1, 0, 1]), np.array([1, -1, 0, 1]), np.array([-1, -1, 0, 1])],
                            9: [np.array([1, 1, 0, 1]), np.array([-1, 1, 0, 1]), np.array([1, -1, 0, 1]), np.array([-1, -1, 0, 1])],}
        self.size = size
    def get_rand_img(root, paths) -> np.ndarray:
        path = paths[random.randrange(0, len(paths))]
        img = cv.imread(os.path.join(root, path), cv.IMREAD_UNCHANGED)
<<<<<<< HEAD
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
=======
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
        if (img.shape[2] == 3):
            img = img[:,:,[2, 1, 0]]
        elif (img.shape[2] == 4):
            img = img[:,:,[2, 1, 0, 3]]
        else:
            assert False
        return img, path
<<<<<<< HEAD

=======
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
    def sign_path2label(self, path:str):
        return self.name2label[path.split("_")[0]]
    def get_bb(self, path: str, id: int, img_shape):
        with open(os.path.join(self.sign_path, "info", path.replace(".png", ".info.json"))) as jf:
            minx = 100000
            maxx = 0
            miny = 100000
            maxy = 0
            mat = json.load(jf)
            mat = np.array(mat)
            for point in self.edge_points[id]:
                point = mat @ point                
                point = point[0:2] / point[3]
                #point[1] *= -1
                point = point * 0.5 + 0.5
                point[0] *= img_shape[1]
                point[1] *= img_shape[0]
                point[1] = img_shape[0] - point[1]
                minx = min(minx, point[0])
                maxx = max(maxx, point[0])
                miny = min(miny, point[1])
                maxy = max(maxy, point[1])
            return int(minx), int(miny), int(maxx), int(maxy)
    def blockout(alpha, bbox):
        if (bbox[2] - bbox[0]) // 4 > 0 and (bbox[3] - bbox[1]) // 4 > 0:
            for i in range(random.randrange(0, 5)):
                w = random.randrange(0, (bbox[2] - bbox[0]) // 4)
                h = random.randrange(0, (bbox[3] - bbox[1]) // 4)
                x = random.randrange(bbox[0], bbox[2] - w)
                y = random.randrange(bbox[1], bbox[3] - h)
                cv.rectangle(alpha, (x, y), (x+w, y+h), 0, -1)
        return alpha
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, idx):
        sign_count = random.randrange(self.scr[0], self.scr[1])
        img, path = fake_dataset.get_rand_img(self.background_path, self.backgrounds)
        boxes = []
        labels = []
        area = []
<<<<<<< HEAD
        x = self.size / min(img.shape[0], img.shape[1])
        img = cv.resize(img, (0, 0), fx = x, fy = x)
        ocluded_box_count = 0
=======
        x = self.size / max(img.shape)
        img = cv.resize(img, (0, 0), fx = x, fy = x)
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
        for i in range(sign_count):
            sign, path = fake_dataset.get_rand_img(os.path.join(self.sign_path, "imgs"), self.signs)
            size = random.randrange(self.sign_size[0], min(img.shape[0]-5, img.shape[1]-5, self.sign_size[1]))
            scaling = size / max(sign.shape)
            sign = cv.resize(sign, dsize=(0, 0), fx=scaling, fy=scaling)
<<<<<<< HEAD
            minx, miny, maxx, maxy = self.get_bb(path, self.sign_path2label(path), sign.shape)
            
            w = sign.shape[1]
            h = sign.shape[0]
            
            x = random.randrange(0, img.shape[1] - sign.shape[1])
            y = random.randrange(0, img.shape[0] - sign.shape[0])
            bbox_img = (x+minx, y+miny, x+maxx, y+maxy)
            ocluded = False
            # Prevent oclusion
            for box in boxes:
                b = box[0], box[1], box[2], box[3]
                if bb_oclusion(b, bbox_img) > 0.1:
                    
                    ocluded = True
                    ocluded_box_count += 1
                    break
            if ocluded:
                continue
            alpha = (sign[:,:,3]/255)
            
            #alpha = fake_dataset.blockout(alpha, bbox)
=======
            bbox = self.get_bb(path, self.sign_path2label(path), sign.shape)
            w = sign.shape[1]
            h = sign.shape[0]
            #if img.shape[1] - sign.shape[1] <= 0 or img.shape[0] - sign.shape[0] <= 0:
            #    continue
            x = random.randrange(0, img.shape[1] - sign.shape[1])
            y = random.randrange(0, img.shape[0] - sign.shape[0])
            alpha = (sign[:,:,3]/255)
            
            alpha = fake_dataset.blockout(alpha, bbox)
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff

            #alpha *= (random.random() * 0.25 + 0.75)
            alpha = np.stack([alpha, alpha, alpha], axis=-1)
            sign = sign[:,:,0:3] * alpha
            img[y:y+h, x:x+w, :] = img[y:y+h, x:x+w, :] * (1-alpha) + sign
            
<<<<<<< HEAD
=======
            minx, miny, maxx, maxy = bbox
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
            boxes.append(np.array([x+minx, y+miny, x+maxx, y+maxy]))
            labels.append(self.sign_path2label(path))
            area.append(w*h)
        target = {"boxes": torch.tensor(np.stack(boxes), dtype=torch.float32), 
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx], dtype=torch.int64),
                  "area": torch.tensor(area, dtype=torch.float32),
<<<<<<< HEAD
                  "iscrowd": torch.tensor([False] * (sign_count - ocluded_box_count)),
                  }
        encode_params = [cv.IMWRITE_JPEG_QUALITY, random.randrange(5, 75)]
        #cv.imdecode()
        buf = cv.imencode(".jpg", img, encode_params)
        #print(buf)
        img = cv.imdecode(buf[1], cv.IMREAD_UNCHANGED)
=======
                  "iscrowd": torch.tensor([False] * sign_count),
                  }
>>>>>>> 00884d9dc7736ceb9037c8a8ad679216526740ff
        img = Image.fromarray(img)
        # = np.transpose(img, (2, 0, 1))
        #img = torch.tensor(img, dtype=torch.float32) / 255
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
if __name__ == "__main__":
    dataset = fake_dataset("blender/out", "backgrounds", 1, (1, 3), 700, (30, 200))
    dataset.__getitem__(0)