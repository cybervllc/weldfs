from collections import defaultdict
import os
import argparse
from sched import scheduler
import time
import copy
from turtle import title

import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import hiddenlayer as hl
import RGBDutils
import matplotlib.pyplot as plt
import pylab
import torchvision.datasets as datasets

default_dir = '/home/andrey/Python-Progs/weldfs/data/'
default_file = 'rgbd.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!
#Parse agruments and setting defaults
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=default_dir)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0001)
    return parser.parse_args()


#Defining the bacjbone for internal ANN
class InternalNet(nn.Module):
    def __init__(self, dims=512):
        super().__init__()
        model = models.resnet18()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1 * 1, dims)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#Defining the two input RGB+Depth (rgb img + hha img) ANN
class RGBDepthNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # RGB branch
        model_rgb = InternalNet()
        self.rgb_convs = model_rgb.features
        num_planes_rgb = 512

        # HHA branch
        model_hha = InternalNet()
        self.hha_convs = model_hha.features
        num_planes_hha = 512

        self.conv = nn.Sequential(
            nn.Conv2d(num_planes_rgb + num_planes_hha, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(512 * 1 * 1, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_rgb = self.rgb_convs(x[0])
        x_hha = self.hha_convs(x[1])

        x = torch.cat((x_rgb, x_hha), 1)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def build_model(num_classes):
    model = RGBDepthNet(num_classes=num_classes)
    # freeze weights
    for module in [model.rgb_convs, model.hha_convs]:
        for param in module.parameters():
            param.requires_grad = False
    return model


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, use_gpu, vis_enable=False):
    # A History object to store metrics
    history1 = hl.History()
    history2 = hl.History()
    history3 = hl.History()
    # A Canvas object to draw the metrics
    canvas1 = hl.Canvas()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        train_loss, train_acc = train_epoch(dataloaders['train'], model, criterion, optimizer, use_gpu)
        print('Train Loss: {:.4f} Accuracy: {:.4f}'.format(train_loss, train_acc))
        history1.log(epoch, loss=train_loss, accuracy=train_acc)
        
        test_loss, test_acc = train_epoch(dataloaders['test'], model, criterion, optimizer, use_gpu)
        print('Test Loss: {:.4f} Accuracy: {:.4f}'.format(test_loss, test_acc))
        history2.log(epoch, loss=test_loss, accuracy=test_acc)
        
        val_loss, val_acc = evaluate_model(dataloaders['val'], model, criterion, use_gpu)
        print('Val Loss: {:.4f} Accuracy: {:.4f}'.format(val_loss, val_acc))
        history3.log(epoch, loss=val_loss, accuracy=val_acc)

        if vis_enable == True: 
            with canvas1:
                canvas1.draw_plot([history1["loss"], history2["loss"], history3['loss']], labels=["Train Loss", "Test Loss", "Val Loss"])
                canvas1.draw_plot([history1["accuracy"], history2["accuracy"], history3['accuracy']], labels=["Train Accuracy", "Test Accuracy", "Val Accuracy"])
                canvas1.draw_summary(history1)
        print()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step(val_loss)
    canvas1.save(default_dir + "Loss_Accuracy_Graphs.jpg")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Accuracy: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def train_epoch(dataloader, model, criterion, optimizer, use_gpu):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for data in dataloader:
        # get the inputs
        inputs_rgb, inputs_hha, labels = data
        if use_gpu:
            inputs_rgb = inputs_rgb.cuda()
            inputs_hha = inputs_hha.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model((inputs_rgb, inputs_hha))
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects / len(dataloader.dataset)
    return loss, acc


def evaluate_model(dataloader, model, criterion, use_gpu):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for data in dataloader:
        # get the inputs
        inputs_rgb, inputs_hha, labels = data
        if use_gpu:
            inputs_rgb = inputs_rgb.cuda()
            inputs_hha = inputs_hha.cuda()
            labels = labels.cuda()

        # forward
        outputs = model((inputs_rgb, inputs_hha))
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects / len(dataloader.dataset)
    return loss, acc


def main():
    #parse input arguments
    args = parse_args()
    use_gpu = torch.cuda.is_available()
    # data augmentation and normalization for training
    RGB_AVG = [0.485, 0.456, 0.406]  # default ImageNet ILSRVC2012
    RGB_STD = [0.229, 0.224, 0.225]  # default ImageNet ILSRVC2012
    DEPTH_AVG = [0.485, 0.456, 0.406]  # default ImageNet ILSRVC2012
    DEPTH_STD = [0.229, 0.224, 0.225]  # default ImageNet ILSRVC2012
    data_transforms = {
        'train': RGBDutils.Compose([
            RGBDutils.Resize(256),
            RGBDutils.CenterCrop(227),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)
        ]),
        'val': RGBDutils.Compose([
            RGBDutils.Resize(256),
            RGBDutils.CenterCrop(227),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)
        ]),
        'test': RGBDutils.Compose([
            RGBDutils.Resize(256),
            RGBDutils.CenterCrop(227),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)
        ]),
    }

    # prepare dataset and dataloaders
    partitions = ['train', 'val', 'test']
    image_datasets = {x: RGBDutils.ImageFolder(os.path.join(args.dataset_dir, x), data_transforms[x])
                      for x in partitions}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'train'), num_workers=4)
                   for x in partitions}
    print(image_datasets)
    print(image_datasets['train'].classes)

    # instantiate the model
    model = build_model(num_classes=len(image_datasets['train'].classes))
    #print(model)
    #print('Model parameters info:', [name for name, param in model.named_parameters() if param.requires_grad])

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    #scheduler = None
   
    # train
    model = train_model(dataloaders, model, criterion, optimizer, scheduler, args.epochs, use_gpu, vis_enable=True)
    # evaluate
    _, train_acc = evaluate_model(dataloaders['train'], model, criterion, use_gpu)
    _, val_acc = evaluate_model(dataloaders['val'], model, criterion, use_gpu)
    _, test_acc = evaluate_model(dataloaders['test'], model, criterion, use_gpu)
    print('Accuracy: Train: {:1.2f}% Val: {:1.2f}% Test: {:1.2f}%'.format(train_acc * 100, val_acc * 100, test_acc * 100))

    # save pretrained model
    torch.save(model.state_dict(), default_dir+default_file)

if __name__ == '__main__':
    main()
