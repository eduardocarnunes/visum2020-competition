import torch
import torchvision
from transforms_6d import get_test_transform
from dataset_6d_Image import Test_Dataset
import csv
from torchvision.ops import nms

NMS_THRESHOLD = 0.1
SAVED_MODEL = '/home/visum/2020-07-09_res50fpn_1x1_aug_v1_2.pth'
DATA_DIR = '/home/master/dataset/test/'

# Load dataset
dataset_test = Test_Dataset(DATA_DIR, transforms=get_test_transform())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
model = torch.load(SAVED_MODEL)
model.to(device)

predictions = list()

for ii, (img, seq, frame) in enumerate(dataset_test):
    if ii%50 == 0:
        print("Processed %d / %d images" % (ii, len(dataset_test)))

    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    scores = prediction[0]['scores'].cpu()

    nms_indices = nms(boxes, scores, NMS_THRESHOLD)

    nms_boxes = boxes[nms_indices].tolist()
    nms_scores = scores[nms_indices].tolist()
        
    # if there are no detections there is no need to include that entry in the predictions
    if len(nms_boxes) > 0:
        for bb, score in zip(nms_boxes, nms_scores):
            if score > 0.4:
                predictions.append([seq, frame, list(bb), score])


with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(['seq', 'frame', 'label', 'score'])
    writer.writerows(predictions)
