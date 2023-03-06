import os
import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob
import random
from torch.utils.data import Dataset
from PIL import Image


########################################################
# Methods for Image DataLoader
#
# 1's = defect
# 0's = no defect
#
#
########################################################

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        # self.transform = None
        
        self.files_defect = np.load(glob.glob(os.path.join(root, "%s_defect" % mode) + "/*.*")[0])
        self.labels_defect = np.ones(np.shape(self.files_defect)[0])
        
        self.files_no_defect = np.load(glob.glob(os.path.join(root, "%s_no_defect" % mode) + "/*.*")[0])
        self.labels_no_defect = np.zeros(np.shape(self.files_no_defect)[0])

        self.combined_data = np.concatenate((self.files_defect, self.files_no_defect))
        self.labels = np.concatenate((self.labels_defect, self.labels_no_defect))
        
        # print('Max ', np.amax(self.combined_data))
        # print('Min ', np.amin(self.combined_data))
        self.mean = self.combined_data.mean()
        self.std = self.combined_data.std()
        
        # self.transform = transforms.Normalize((self.mean), (self.std))
        
    def __getitem__(self, index):
        image = torch.from_numpy(self.combined_data[index])
        label = self.labels[index]
        image = image.unsqueeze(0)
        if self.transform != None:
            image = self.transform(image)
        # print('Image shape ', np.shape(image))
        return image, label

    def __len__(self):
        return len(self.combined_data)


class ExperimentalDataset(Dataset):
    def __init__(self, root, transforms_=None):
        # self.transform = None
        self.transform = transforms.Compose(transforms_)

        
        self.files_defect = np.load(glob.glob(os.path.join(root, "exp_defect") + "/*.*")[0])
        self.labels_defect = np.ones(np.shape(self.files_defect)[0])
        
        self.files_no_defect = np.load(glob.glob(os.path.join(root, "exp_no_defect") + "/*.*")[0])
        self.labels_no_defect = np.zeros(np.shape(self.files_no_defect)[0])

        self.combined_data = np.concatenate((self.files_defect, self.files_no_defect))
        self.labels = np.concatenate((self.labels_defect, self.labels_no_defect))
                
        # print('Max ', np.amax(self.combined_data))
        # print('Min ', np.amin(self.combined_data))
        
        self.mean = self.combined_data.mean()
        self.std = self.combined_data.std()
        
        # self.transform = None#transforms.Normalize((self.mean), (self.std))
        
        

    def __getitem__(self, index):
        image = (self.combined_data[index])
        image = torch.from_numpy(self.combined_data[index])
        # image = image.float()
        label = self.labels[index]
        image = image.unsqueeze(0)
        if self.transform != None:
            image = self.transform(image)
        # print('Image shape ', np.shape(image))
        return image, label

    def __len__(self):
        return len(self.combined_data)

