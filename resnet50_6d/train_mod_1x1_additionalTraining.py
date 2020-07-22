import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils
from dataset_6d_Image import Dataset
from transforms_6d import get_transform

# custom imports
from torch.utils.tensorboard import SummaryWriter
from split_data import get_sequence_stats, split_data, seq_indices_to_frame_indices
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

DATA_DIR = '/home/master/dataset/train/'
LOAD_MODEL = '/home/visum/2020-07-08_res50fpn_1x1_19.pth'
SAVE_MODEL = ('2020-07-08_res50fpn_1x1_additionalTraining')

print(f'Results are stored to runs/{SAVE_MODEL}')
# Summary writer for tensorboard
writer = SummaryWriter(f'runs/{SAVE_MODEL}')

# load a pre-trained model for classification and return
# only the features

# resnet_net = torchvision.models.resnet50(pretrained=True)
# modules = list(resnet_net.children())[:-2]
# backbone = torch.nn.Sequential(*modules)

#backbone = resnet_fpn_backbone('resnet50', pretrained=True)

#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 12img
# so we need to add it here
#backbone.out_channels = 2048 #, 512, 1024, 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 4 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each featureimg
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=["0"], output_size=7, sampling_ratio=2
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,image_mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

print(model)

# copy weights of the pretrained model
copied_weight = model.backbone.body.conv1.weight.clone().detach()
# input = torch.cat(x_image, x_depth, dim=1) # RGBD input

# do an extra layer with 4-dimensions
model.backbone.body.conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    model.backbone.body.conv1.weight[:, :3] = copied_weight
    model.backbone.body.conv1.weight[:, 3:] = copied_weight
   # model.backbone.body.conv1.weight.requires_grad_()

model = torch.load(LOAD_MODEL)
# TODO check if gradients are computed with respect to new weights

# put the pieces together inside a FasterRCNN model
# one class for fish, other for the backgroud
# model = FasterRCNN(
#     backbone,
#     num_classes=2,
#     rpn_anchor_generator=anchor_generator,
#     box_roi_pool=roi_pooler,
#     min_size=300, max_size=300
# )

# See the model architecture
print(model)


# use our dataset and defined transformations
dataset = Dataset(DATA_DIR, transforms=get_transform(train=True))
dataset_val = Dataset(DATA_DIR, transforms=get_transform(train=False))
dataset_test = Dataset(DATA_DIR, transforms=get_transform(train=False))

# split the dataset into train and validation sets
torch.manual_seed(1)
# get similar distributated train, val and test set
sequences, sequenceStats = get_sequence_stats()
training_seq_indices, validation_seq_indices, testing_seq_indices = split_data(sequenceStats)

training_indices = seq_indices_to_frame_indices(training_seq_indices)  #dataset.ann = load_labels()
validation_indices = seq_indices_to_frame_indices(validation_seq_indices)  #dataset.ann = load_labels()
testing_indices = seq_indices_to_frame_indices(testing_seq_indices)  #dataset.ann = load_labels()

# not needed anymore indices = torch.randperm(len(dataset)).tolist()
dataset_sub = torch.utils.data.Subset(dataset, training_indices)
dataset_val_sub = torch.utils.data.Subset(dataset_val, validation_indices)
dataset_test_sub = torch.utils.data.Subset(dataset_test, testing_indices)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_sub, batch_size=6, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val_sub, batch_size=6, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model.to(device)

# define an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

num_epochs = 20

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    epoch_loss = train_one_epoch(model, optimizer, data_loader,
                                    device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the validation dataset
    mAP, AP = evaluate(model, data_loader_val, dataset_val, device)

    writer.add_scalar('training loss', epoch_loss, epoch)
    writer.add_scalar('mAP', mAP, epoch)

    # save model per epoch
    file_name_model_epoch = SAVE_MODEL + '_' + str(epoch) + '.pth'
    torch.save(model, file_name_model_epoch)
print(f'Testseq to remember: {testing_seq_indices}')
writer.close()
torch.save(model, SAVE_MODEL)
