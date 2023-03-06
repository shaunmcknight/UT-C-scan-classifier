import numpy as np
import itertools
import time
import datetime
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
from torchinfo import summary

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import ImageGrid

from IPython.display import clear_output
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


from PIL import Image
import matplotlib.image as mpimg

from utils import *
from classifier import *


##############################################
# Defining all hyperparameters
##############################################


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


##############################################
# Final Training Function
##############################################

def train(
    train_dataloader,
    n_epochs,
    criterion,
    optimizer,
    Tensor,
    early_stop,
    Net
):
    losses = []
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = Net(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    np.mean(loss.item()*hp.batch_size),
                    time_left,
                )
            )

            losses.append(np.mean(loss.item()*hp.batch_size))

            prev_time = time.time()

        if (np.mean(loss.item()*hp.batch_size)) < early_stop:
            break

    print('Finished Training')

    plt.figure()
    plt.plot(losses)
    plt.title('Network Losses (Batch average)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def test(dataloader, description, disp_CM, Net, Tensor):
    true_list = []
    pred_list = []
    Net.eval()
    Net.cpu()  # cuda()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                images = images.cpu()  # cuda()
                labels = labels.cpu()  # cuda()

            for i in range(len(labels.numpy())):
                true_list.append(labels.numpy()[i])

            output = Net(images)
            output = torch.sigmoid(output)
            pred_tag = torch.round(output)
            [pred_list.append(pred_tag[i]) for i in range(
                len(pred_tag.squeeze().cpu().numpy()))]

    pred_list = [a.squeeze().tolist() for a in pred_list]

    true_list = np.array(true_list)
    pred_liSt = np.array(pred_list)

    correct = np.sum(true_list == pred_list)
    total = np.shape(true_list)

    print('')
    print('~~~~~~~~~~~~~~~~~')
    print(description)
    print('Prediciton Accuracy: ', (correct/total)*100)

    print('Confusion matrix || {}'.format(description))
    cm = confusion_matrix(true_list, pred_list)
    print(cm)

    if disp_CM == True:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['No defect', 'Defect'])
        disp.plot()
        disp.ax_.set_title(description)

    precision, recall, f_score, support = precision_recall_fscore_support(
        true_list, pred_list)

    print('Precision ', precision[1])
    print('Recall ', recall[1])
    print('F score ', f_score[1])

    return true_list, pred_list, f_score[1]


def showFailures(true_list, pred_list):
    list_incorrect_defects = []
    list_incorrect_clean = []
    for i, item in enumerate(pred_list):
        if item != true_list[i]:
            if item == 1:
                list_incorrect_defects.append(
                    experimental_dataloader.dataset[i][0].squeeze())
            elif item == 0:
                list_incorrect_clean.append(
                    experimental_dataloader.dataset[i][0].squeeze())
            else:
                print('ERROR')

    if len(list_incorrect_defects) > 0:
        plotGrid(list_incorrect_defects,
                 'Grid of incorrect defect predicitions')

    if len(list_incorrect_clean) > 0:
        plotGrid(list_incorrect_clean,
                 'Grid of incorrect clean predicitions')


def plotGrid(img_list, title):
    grid_dim = math.floor((len(img_list))**(1/2))
    fig = plt.figure()
    plt.suptitle(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     # creates 2x2 grid of axes
                     nrows_ncols=(grid_dim, math.ceil(len(img_list)/grid_dim)),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, img_list):
        ax.imshow(im)
        ax.axis('off')

""" Hiding for Chris HP tuning code
# hp = Hyperparameters(
#     epoch=0,
#     n_epochs=250,
#     dataset_train_mode="train",
#     dataset_test_mode="test",
#     batch_size=64,
#     lr=0.1,
#     momentum=0.9,
#     img_size=64,
#     channels=1,
#     early_stop=3
# )


# root_path = r"C:/Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\civa\noised"  # "_with_noise"

# ##############################################
# # Defining Image Transforms and data_loaders
# ##############################################

# transforms_ = [
#     # transforms.ToTensor(),
#     # transforms.Normalize((0.5), (0.5)),
# ]

# train_dataloader = DataLoader(
#     ImageDataset(root_path, mode=hp.dataset_train_mode,
#                  transforms_=transforms_),
#     batch_size=hp.batch_size,
#     shuffle=True,
#     # num_workers=1,
# )
# test_dataloader = DataLoader(
#     ImageDataset(root_path, mode=hp.dataset_test_mode,
#                  transforms_=transforms_),
#     batch_size=16,
#     shuffle=False,
#     # num_workers=1,
# )

# experimental_dataloader = DataLoader(
#     ExperimentalDataset(root_path, transforms_=transforms_),
#     batch_size=16,
#     shuffle=False,
#     # num_workers=1,
# )


# ##############################################
# # SAMPLING IMAGES
# ##############################################

# test_data = next(iter(train_dataloader))
# plt.figure()
# plt.imshow(test_data[0][0].squeeze(), cmap='gray')
# # plt.clim(0,1)
# plt.colorbar()

# test_data = next(iter(test_dataloader))
# plt.figure()
# plt.imshow(test_data[0][0].squeeze(), cmap='gray')
# plt.colorbar()

# test_data = next(iter(experimental_dataloader))
# plt.figure()
# plt.imshow(test_data[0][0].squeeze(), cmap='gray')
# # plt.clim(0,1)
# plt.colorbar()


# ##############################################
# # SETUP, LOSS, INITIALIZE MODELS and OPTIMISERS
# ##############################################

# input_shape = (hp.channels, hp.img_size, hp.img_size)

# # Net = Network(input_shape)


# Net = DynamicNetwork(input_shape, conv_layers=4,
#                      kernel_size=3, out_channel_ratio=3, FC_layers=2)


# # Network summary info
# print('Network')

# # print(Net)
# print(summary(Net.float(), input_size=(hp.batch_size, 1, 64, 64)))

# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(Net.parameters(), lr=hp.lr, momentum=hp.momentum)


# Net = Net.double()

# # CUDA operations

# cuda = True if torch.cuda.is_available() else False
# print("Using CUDA" if cuda else "Not using CUDA")

# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# if cuda:
#     Net = Net.cuda()
#     criterion = criterion.cuda()


# ##############################################
# # Execute the Final Training Function
# ##############################################


# train(
#     train_dataloader=train_dataloader,
#     n_epochs=hp.n_epochs,
#     criterion=criterion,
#     optimizer=optimizer,
#     Tensor=Tensor,
#     early_stop=hp.early_stop
# )
# test(test_dataloader, 'Validation Test', disp_CM=False)
# true_list, pred_list = test(experimental_dataloader,
#                             'Experimental Test', disp_CM=True)
# showFailures(true_list, pred_list)

"""

# function that takes in HPs and gives out F score

def chris(HP, cuda_device):
    
    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]
    
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
 
    global hp

    hp = Hyperparameters(
        epoch=0,
        n_epochs=HP['n_epochs'],
        dataset_train_mode="train",
        dataset_test_mode="test",
        batch_size=HP['batch_size'],
        lr=HP['lr'],
        momentum=HP['momentum'],
        img_size=64,
        channels=1,
        early_stop=HP['early_stop']
    )

        
    root_path = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\civa\noised"  # "_with_noise"
    
    ##############################################
    # Defining Image Transforms and data_loaders
    ##############################################
    
    transforms_ = [
        # transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5)),
    ]
    
    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode,
                     transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    test_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_test_mode,
                     transforms_=transforms_),
        batch_size=16,
        shuffle=False,
        # num_workers=1,
    )
    
    experimental_dataloader = DataLoader(
        ExperimentalDataset(root_path, transforms_=transforms_),
        batch_size=16,
        shuffle=False,
        # num_workers=1,
    )
    
    
    ##############################################
    # SAMPLING IMAGES
    ##############################################
    
    test_data = next(iter(train_dataloader))
    plt.figure()
    plt.imshow(test_data[0][0].squeeze(), cmap='gray')
    # plt.clim(0,1)
    plt.colorbar()
    
    test_data = next(iter(test_dataloader))
    plt.figure()
    plt.imshow(test_data[0][0].squeeze(), cmap='gray')
    plt.colorbar()
    
    test_data = next(iter(experimental_dataloader))
    plt.figure()
    plt.imshow(test_data[0][0].squeeze(), cmap='gray')
    # plt.clim(0,1)
    plt.colorbar()
    
    
    ##############################################
    # SETUP, LOSS, INITIALIZE MODELS and OPTIMISERS
    ##############################################
    
    input_shape = (hp.channels, hp.img_size, hp.img_size)
    
    # Net = Network(input_shape)
    
    
    # Net = DynamicNetwork(input_shape, conv_layers=4,
    #                      kernel_size=3, out_channel_ratio=3, FC_layers=2)
    Net = DynamicNetwork(input_shape, conv_layers=HP['conv_layers'],
                         kernel_size=3, out_channel_ratio=HP['out_channel_ratio'],
                         FC_layers=HP['FC_layers'])
    
    
    # Network summary info
    print('Network')
    
    # print(Net)
    print(summary(Net.float(), input_size=(hp.batch_size, 1, 64, 64)))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(Net.parameters(), lr=hp.lr, momentum=hp.momentum)
    
    
    Net = Net.double()
    
    # CUDA operations
    
    cuda = True if torch.cuda.is_available() else False
    print("Using CUDA" if cuda else "Not using CUDA")
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    if cuda:
        Net = Net.cuda()
        criterion = criterion.cuda()
    
    
    ##############################################
    # Execute the Final Training Function
    ##############################################
    
    
    train(
        train_dataloader=train_dataloader,
        n_epochs=hp.n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        Tensor=Tensor,
        early_stop=hp.early_stop,
        Net = Net
    )
    test(test_dataloader, 'Validation Test', disp_CM=False, Net = Net, Tensor=Tensor)
    true_list, pred_list, f_score = test(experimental_dataloader,
                                'Experimental Test', disp_CM=True, Net = Net, Tensor=Tensor)
    # showFailures(true_list, pred_list)
    
    return f_score

    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]

# Value limits:
# n_epochs: something reasonable 50 - 1000? (250 seems to work well as a ball aprk)
# 'batch_size': [2,4,8,16,32,64,128,256]
# 'lr': 0 - 1
# 'momentum': 0 - 1
# 'early_stop': 0 - 5 or 10 seems reasonable?
# 'conv_layers': 1 - 6
# 'out_channel_ratio': 1 - this is memory dependant as model size scales with out_ratio**layers maybe keep to 1-3 or 4
# 'FC_layers': 1-6 seems reasonable 1 or 2 seems to works well

HP = {'n_epochs': 100,
      'batch_size': 64,
      'lr': 0.9,
      'momentum': 0.9,
      'early_stop': 2,
      'conv_layers': 3,
      'out_channel_ratio': 3,
      'FC_layers':7
      }


# add in cuda device

print(chris(HP, None))