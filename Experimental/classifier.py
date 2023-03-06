import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset
from PIL import Image


##############################################
# CNN
##############################################

    
class Network(nn.Module):

    def __init__(self, input_shape):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(5, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        # self.conv3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(2560, 1024)
        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        # print(x.shape)
        x = x.view(-1, 20*16*16)
        # x = x.view(-1, 40*8*8)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        return (x)
    