
'''
  Since we are using built-in datasets, we don't really
  need to customize datasets and dataloader that much.
'''
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Caltech256
from torchvision.transforms import Compose, Resize, Lambda,\
    RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, ToTensor

#transform designed for input image
image_transform = Compose([
    ToTensor(),
    #some pictures only contain one channel, expand them to three
    Lambda(lambda x: x if x.shape[0]==3 else torch.cat([x,x,x])),
    Resize(224),
    RandomCrop(224),
    RandomHorizontalFlip(),
    RandomVerticalFlip()
])

#transform designed for target
def target_transform(label):
    re = torch.zeros((257))
    re[label] = 1
    return re

#getter method
def get_data_loader(batch_size=16, root='/dataset'):

    caltech = Caltech256(root, image_transform,\
                         target_transform, download=True)
    re = DataLoader(caltech, batch_size, shuffle=True)

    return re