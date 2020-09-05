import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os
import copy
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

TRAIN_DATASET_PATH = '../dataset/train_data'
IMG_SIZE = (224, 224)  # 原值(512, 512)
BATCH_SIZE = 14  # 原值20
LR = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 预训练模型
path_state_dict_50 = r"D:\机器学习\PycharmProjects3.7\论文阅读\05CV-baseline\pre_train_model\resnet50-19c8e357.pth"


# ============================ step 1/5 数据 ============================
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_fns, label_dict, data_transforms):
        self.image_fns = image_fns
        self.label_dict = label_dict
        self.transforms = data_transforms

    def __getitem__(self, index):
        # label = self.label_dict[image_fns[index].split('/')[-2]] # linux 系统
        label = self.label_dict[image_fns[index].split('\\')[-2]]  # windows 系统
        image = Image.open(image_fns[index]).convert("RGB")
        image = self.transforms(image)

        return image, label  # , image_fns[index]

    def __len__(self):
        return len(self.image_fns)


image_fns = glob(os.path.join(TRAIN_DATASET_PATH, '*', '*.*'))
# label_names = [s.split('/')[-2] for s in image_fns]  # linux 系统
label_names = [s.split('\\')[-2] for s in image_fns]  # windows 系统

unique_labels = list(set(label_names))
unique_labels.sort()
id_labels = {_id: name for name, _id in enumerate(unique_labels)}

NUM_CLASSES = len(unique_labels)
print("NUM_CLASSES:", NUM_CLASSES)

# imagenet数据集 统计值
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    [
        transforms.Resize(256),  # 缩放图片尺寸，保证最小边到256
        transforms.CenterCrop(IMG_SIZE),  # 中心裁剪到224*224--正方形
        transforms.RandomRotation((-15, 15)),  # 随机旋转角度
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 颜色
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)])

train_fns, val_fns = train_test_split(image_fns, test_size=0.1, shuffle=True)

train_dataset = ImageDataset(train_fns, id_labels, train_transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=2)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)
val_dataset = ImageDataset(val_fns, id_labels, val_transform)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=True)
datalaoders_dict = {'train': train_loader, 'val': val_loader}


# ============================ step 2/5 模型 ============================

def get_EfficientNet(device, num_classes=1000, vis_model=True):
    """
    创建resnet50模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = EfficientNet.from_pretrained('efficientnet-b4',
                                         weights_path="pre_train_model/efficientnet-b4-6ed6700e.pth",
                                         num_classes=num_classes)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    #model.to(device)
    return model


model = get_EfficientNet(device, num_classes=NUM_CLASSES, vis_model=True)
model = model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 训练
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phaseSS
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], ncols=80):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'models/resnet50_224_best.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_net = train_model(model, datalaoders_dict, criterion, optimizer, exp_lr_scheduler, num_epochs=20)
# torch.save(model_net.state_dict(), 'models/resnet50_224_best.pth')
