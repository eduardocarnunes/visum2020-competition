import random
import torchvision

import imgaug.augmenters as iaa

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

# Data augmentation
def get_transform(train):
    transforms = iaa.Identity()
    if train:
        transforms = iaa.SomeOf((0,None), [
            iaa.Affine(scale=(1.0,1.5)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            iaa.Fliplr(),
            iaa.MotionBlur(k=(3,7))
        ])
    return transforms

class ToTensor(object):
    def __call__(self, image, prev_img, target):
        image = F.to_tensor(image)
        prev_img = F.to_tensor(prev_img)
        return image, prev_img, target

# transform for images only (no labels)
def get_test_transform():
    # in case you want to insert some transformation in here
    return iaa.Identity()

from PIL import Image, ImageDraw
from dataset_6d_Image_aug import boxes2IaaBoxes,iaaBoxes2Boxes
import numpy as np

if __name__ == '__main__':
    transforms=get_transform(train=True)
    with Image.open('/home/master/dataset/train/seq009/img0.jpg').convert('RGB') as im:
        with Image.open('/home/master/dataset/train/seq009/img1.jpg').convert('RGB') as prevIm:
            for i in range(0,100):
                transforms_det = transforms.to_deterministic()
                numpyPrevImg = np.asarray(prevIm)
                numpyImg = np.asarray(im)
                boxes = [(190,163,394,429)]
                augNumpyImg, iaaBoxes = transforms_det.augment(image=numpyImg,bounding_boxes=boxes2IaaBoxes(boxes))
                augNumpyPrevImg, _ = transforms_det.augment(image=numpyPrevImg,bounding_boxes=[])

                augImg = Image.fromarray(augNumpyImg)
                augPrevImg = Image.fromarray(augNumpyPrevImg)
                draw = ImageDraw.Draw(augImg)
                boxes = iaaBoxes2Boxes(iaaBoxes)
                (x1,y1,x2,y2) = boxes[0]

                draw.rectangle([(x1,y1),(x2,y2)])
                augImg.save('flatfish' + str(i) + '.png')
                augPrevImg.save('flatfish' + str(i) + '_prev.png')
                print(i)