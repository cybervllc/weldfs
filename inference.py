
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torchvision.utils as utils
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from torchviz import make_dot
import hiddenlayer as hl
import RGBDutils
import torchvision
import os
import argparse

#Input image normalization constants
RGB_AVG = [0.485, 0.456, 0.406]  # default ImageNet ILSRVC2012
RGB_STD = [0.229, 0.224, 0.225]  # default ImageNet ILSRVC2012

default_dir = '/home/andrey/Python-Progs/weldfs/data/'
default_file = 'rgbd.pt'

# ANN Internal as Backbone (resnet18)
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


#RGBD Class for two input ANN: RGB(image) and HHA(depth image)
class RGBDepthNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # RGB branch
        model_rgb = InternalNet()
        self.rgb_convs = model_rgb.features
        num_planes_rgb = 512

        # Depth branch
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

# Visualization of the geature map
def visualize_activation_maps(input, model, name):
    I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
    img = I.permute((1, 2, 0)).cpu().numpy()

    conv_results = []
    x = input
    for idx, operation in enumerate(model.rgb_convs): #feature maps enumeration
        x = operation(x)
        if idx in {0, 1, 2, 3, 4, 5, 6, 7, 8}: #conv layers
            conv_results.append(x)
    
    for i in range(8):
        conv_result = conv_results[i]
        N, C, H, W = conv_result.size()
        mean_acti_map = torch.mean(conv_result, 1, True)
        mean_acti_map = F.interpolate(mean_acti_map, size=[224,224], mode='bilinear', align_corners=False)
        map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)
        map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
        map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
        map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
        map_grid = np.float32(map_grid) / 255
        visual_acti_map = 0.6 * img + 0.4 * map_grid
        tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)
        file_name_visual_acti_map = default_dir + name + 'conv{}_activation_map.jpg'.format(i+1)
        utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)
    return 0

def main():
    
    #Setting inference device to cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!
    #Defining the model class (RGBDNet with numclasses)
    model = RGBDepthNet(num_classes=4)
    #Loading the pretrained model
    model.load_state_dict(torch.load(default_dir+default_file))
    #Setting loaded model to cuda device
    model.to(device)
    print('Model was loaded sucessfully an below is a model configuration.')
    print(model)
    print('-'*100)
    #model.eval()

    # data transforms, for pre-processing the input testing image before feeding into the net
    image_transforms = transforms.Compose([
         transforms.Resize((255)),  
         transforms.CenterCrop(224),
         transforms.ToTensor(),     
         transforms.Normalize(RGB_AVG, RGB_STD)
    ])

# data augmentation and normalization for training
    DEPTH_AVG = [0.485, 0.456, 0.406]  # default ImageNet ILSRVC2012
    DEPTH_STD = [0.229, 0.224, 0.225]  # default ImageNet ILSRVC2012
    data_transforms = {
        'train': RGBDutils.Compose([
            RGBDutils.Resize(227),
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
    local_batch = 100
    #loading the data for visualizaation
    partitions = ['train', 'val', 'test']
    image_datasets = {x: RGBDutils.ImageFolder(os.path.join(default_dir, x), data_transforms[x])
                      for x in partitions}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=local_batch, num_workers=4)
                   for x in partitions}
    print('Images datasets info: ')
    print(image_datasets)
    print('-'*100)
    print('Classes names info: ', image_datasets['train'].classes)
    for data in dataloaders['train']:
        # get the inputs
        inputs_rgb, inputs_hha, labels = data

    print("Classes labels for the samples info: ",local_batch, labels)
    print('-'*100)
    outRGB = torchvision.utils.make_grid(inputs_rgb)
    outDepth = torchvision.utils.make_grid(inputs_hha)
    RGBDutils.imshow(outRGB,outDepth, title='Images Batch From Input Dataset')
    
    #Loading images: RGB abd Depth
    img1 = Image.open(default_dir+'val/rgb/class1/7.png')
    img2 = Image.open(default_dir+'val/depth/class1/7.png')
   
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('RGB Image')
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Depth Image')
    plt.show()

    #Info about images and corresponded tensors
    print("original image's 1 shape: " + str(img1.size))
    # pre-process the input 1
    transformed_img1 = image_transforms(img1)
    print("transformed image's 1 shape: " + str(transformed_img1.shape))
        # form a batch with only one image
    batch_img1 = torch.unsqueeze(transformed_img1, 0)
    print("image batch's 1 shape: " + str(batch_img1.shape))
    print('-'*100)
    print("original image's 2 shape: " + str(img2.size))
    # pre-process the input 2
    transformed_img2 = image_transforms(img2)
    print("transformed image's 2 shape: " + str(transformed_img2.shape))
        # form a batch with only one image
    batch_img2 = torch.unsqueeze(transformed_img2, 0)
    print("image batch's shape: " + str(batch_img2.shape))
    print('-'*100)
    #Obtaining the output of the model
    output = model((batch_img1.cuda(), batch_img2.cuda()))
    
    print("Output vector's shape: " + str(output.shape))
    print("Output info:")
    print(output)
    print('-'*100)
    # map the class no. to the corresponding label
    with open(default_dir+'class_names_rgbdepth.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]
    print('-'*100)
    print("Classes info:")
    for i in range(4):
        print("Class " + str(i+1) + ": " + str(classes[i]))
    print('-'*100)
    #visualization of the activation maps of input images
    visualize_activation_maps(batch_img1.cuda(), model, "img1")
    visualize_activation_maps(batch_img2.cuda(), model, "img2")

    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=1)[0] * 100.0
    # obtain the classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:4]]
    print("Set 4 classes. Inference info: input RGB+Depth images belongs to")
    for i in range(4):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))
    print('-'*100)
   #Uncomment to show and save ANN Visual Model
   # ann_graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=True)
   # ann_graph.render (default_dir + 'RGBDepthNet_full', view = False) 
   # ann_graph.view()

   # ann_graph = make_dot(output, params=dict(model.named_parameters()))
   # ann_graph.render (default_dir + 'RGBDepthNet_lite', view = False) 
   # ann_graph.view()
    print(len(inputs_rgb))
if __name__ == "__main__":
    main()