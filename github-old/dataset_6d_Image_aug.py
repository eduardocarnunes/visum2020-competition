import os
import numpy as np
import torch
from PIL import Image
from glob import glob
import re
import csv

import imgaug.augmenters as iaa
from torchvision.transforms import functional as F

import imgaug as ia
def boxes2IaaBoxes(boxes):
    iaaBoxes = []
    for (x1,y1,x2,y2) in boxes:
        iaaBoxes.append(ia.BoundingBox(x1,y1,x2,y2))
    return iaaBoxes

def iaaBoxes2Boxes(iaaBoxes):
    boxes = []
    for iaaBox in iaaBoxes:
        boxes.append((iaaBox.x1,iaaBox.y1,iaaBox.x2,iaaBox.y2))
    return boxes

# TODO: Return prevImg on __getitem__
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=iaa.Identity()):
        self.root = root
        self.transforms = transforms

        # load all image files
        self.ann_file = os.path.join(root, 'labels.csv')
        self.ann = load_labels(self.ann_file)
        self.n_imgs = len(self.ann)

    def __getitem__(self, idx):
        # load images and masks
        seq = self.ann[idx][0]
        frame = self.ann[idx][1]

        img_path = self.root + 'seq' + str(seq) + '/img' + str(frame) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        
        prev_frame = int(frame) - 1
        
        # if it's the start of the sequence give it the following instead of privious frame
        if int(frame) == 0:
            prev_frame = 1

        prev_img_path = self.root + 'seq' + str(seq) + '/img' + str(prev_frame) + '.jpg'
        prev_img = Image.open(prev_img_path).convert('RGB')

        # get bounding box coordinates for each mask
        boxes = [x for x in self.ann[idx][2]]
        n_objs = len(boxes)

        # Transform the images
        img = np.asarray(img)
        prev_img = np.asarray(prev_img)

        transforms_det = self.transforms.to_deterministic()
        img, iaaBoxes = transforms_det.augment(image=img,bounding_boxes=boxes2IaaBoxes(boxes))
        prev_img, _ = transforms_det.augment(image=prev_img,bounding_boxes=[])

        boxes = iaaBoxes2Boxes(iaaBoxes)

        if n_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((n_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((n_objs,), dtype=torch.int64)
        
        # Concatenate the images
        concat_img = torch.cat((F.to_tensor(img), F.to_tensor(prev_img)), dim=0)
        
        return concat_img, target, (seq, frame)

    def __len__(self):
        return self.n_imgs

class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=iaa.Identity()):
        self.root = root
        self.transforms = transforms
        pattern = re.compile('img(.*).jpg')
        self.imgs = list()
        sequences = glob(os.path.join(root, "seq*"))
        for seq in sequences:
            seq_num = seq[-3::]
            img_files = glob(os.path.join(seq, "img*.jpg"))
            for img_f in img_files:

                img_num = pattern.search(os.path.basename(img_f)).group(1)
                self.imgs.append((img_f, seq_num, img_num))
        self.imgs.sort(key=lambda x: (int(x[1]),int(x[2])))

    def __getitem__(self, idx):
        img_file = self.imgs[idx][0]
        seq = self.imgs[idx][1]
        frame = self.imgs[idx][2]

        img = Image.open(img_file).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, seq, frame

    def __len__(self):
        return len(self.imgs)

def load_labels(path_to_csv):
    labels = []
    with open(path_to_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i==0: # header
                continue
            if len(row) == 3:
                boxes = eval(row[2])
            else:
                boxes = []
            labels.append([row[0], row[1], boxes])
    labels.sort(key=lambda x: (int(x[0]), int(x[1])))
    print(labels)

    """
    with open(path_to_csv, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.split(';')            
            if len(line)==3:
                boxes = eval(line[2])
            else:
                boxes = []
            labels.append([line[0], line[1], boxes])

            box = line[2][2:-3]
                for char in '()':
                    box = box.replace(char,'')            
                box = box.split(',')
                bboxs = []
                for bb in range(len(box)//4):
                    bbox = [int(x) for x in box[4*bb:4*bb+4]]
                    bboxs.append(bbox)
    """
    return labels

from transforms_mod import get_transform
if __name__ == '__main__':
     DATA_DIR = '/home/master/dataset/train/'
     dataset = Dataset(DATA_DIR, transforms=get_transform(train=True))
     resu = dataset.__getitem__(2)
     print(resu)