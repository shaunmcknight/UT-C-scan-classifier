import numpy as np
import itertools
import time
import datetime

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
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


from PIL import Image
import matplotlib.image as mpimg

from utils import *
from classifier import *


print("")
cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")

""" So generally both torch.Tensor and torch.cuda.Tensor are equivalent. You can do everything you like with them both.
The key difference is just that torch.Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
Of course operations on a CPU Tensor are computed with CPU while operations for the GPU / CUDA Tensor are computed on GPU. """
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


##############################################
# Defining all hyperparameters
##############################################




class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=150,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=64,
    lr=0.001,
    momentum = 0.9,
    n_cpu=8,
    img_size=64,
    channels=1,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=6,
    lambda_id=5.0,
)

##############################################
# Setting Root Path for Google Drive or Kaggle
##############################################

# Root Path for Google Drive
root_path = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\synthetic\GAN"
root_path = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\experimental"

# Root Path for Kaggle
# root_path = '../input/summer2winter-yosemite'


########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(size, size))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""  The reason for doing "np.transpose(npimg, (1, 2, 0))"
PyTorch modules processing image data expect tensors in the format C × H × W.
Whereas PILLow and Matplotlib expect image arrays in the format H × W × C
so to use them with matplotlib you need to reshape it
to put the channels as the last dimension:
I could have used permute() method as well like below
plt.imshow(pytorch_tensor_image.permute(1, 2, 0))
"""


def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
    imgs = next(iter(val_dataloader))
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()
    
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, [im1, im2, im3, im4, im5, im6]):
        # Iterating over the grid returns the Axes.
        ax.imshow(np.transpose(im, (1, 2, 0)).squeeze())
    
    plt.show()


##############################################
# Defining Image Transforms to apply
##############################################
transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
]

experimental_dataloader = DataLoader(
    ExperimentalDataset(root_path, transforms_ = transforms_),
    batch_size=16,
    shuffle=True,
    # num_workers=1,
)

split_val, split_test = round(len(experimental_dataloader.dataset)*0.8), round(len(experimental_dataloader.dataset))-round(len(experimental_dataloader.dataset)*0.8)

valid_exp, test_exp = torch.utils.data.random_split(experimental_dataloader.dataset, (split_val, split_test))

experimental_dataloader_train = DataLoader(
    valid_exp,
    batch_size=64,
    shuffle=True,
    # num_workers=1,
)

experimental_dataloader_test = DataLoader(
    test_exp,
    batch_size=64,
    shuffle=True,
    # num_workers=1,
)

# print('train max, min: ', ImageDataset.get_max(), train_dataloader.get_min())

test = next(iter(experimental_dataloader_test))
plt.figure()
plt.imshow(test[0][0].squeeze(), cmap = 'gray')
plt.colorbar()

test = next(iter(experimental_dataloader_train))
plt.figure()
plt.imshow(test[0][0].squeeze(), cmap = 'gray')
plt.colorbar()








# Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)

# print('GEN Network')
# print(Gen_AB)
# print('GEN test')

# print(summary(Gen_AB, input_size=(1,1,64,64)))




def train(
    train_dataloader,
    n_epochs,
    criterion,
    optimizer,
    Tensor,
    sample_interval,
):
    

    running_loss = 0
    losses = []
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        
        for i, batch in enumerate(train_dataloader):

            images, labels = batch
            # print('shape', np.shape(images))
            images.type(Tensor)
            labels.type(Tensor)
            # print('shape labels', np.shape(labels))

            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = Net(images)
            # print('shape output', np.shape(outputs))
            # print('outputs shape ', np.shape(outputs))
        
            # print('labels shape ', np.shape(labels))
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                # running_loss = 0.0
    
          
            ##################
            #  Log Progress
            ##################

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

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
                , end="\r"
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                clear_output()
                # plot_output(save_img_samples(batches_done), 30, 40)
                
            losses.append(np.mean(loss.item()*hp.batch_size))
        
        if (np.mean(loss.item()*hp.batch_size)) < 2.5:
            break

    print('Finished Training')
    
    plt.figure()
    plt.plot(losses)
    plt.title('Network Losses (Batch average)')
    plt.show()
    

def test():
    print('')
    print('Test Val')
    true_list = []
    pred_list = []
    Net.eval()
    Net.cpu() #cuda()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)
            # print('shape labels', np.shape(labels))
            # print(np.shape(labels))

            if torch.cuda.is_available():
                images = images.cpu() # cuda()
                labels = labels.cpu() #cuda()
                
            for i in range(len(labels.numpy())):
                true_list.append(labels.numpy()[i]) 
            # true_list.append(labels.numpy())
            output = Net(images)
            output = torch.sigmoid(output)
            pred_tag = torch.round(output)
            [pred_list.append(pred_tag[i]) for i in range(len(pred_tag.squeeze().cpu().numpy()))]

            # pred_list.append(pred_tag.squeeze().cpu().numpy())
 
    pred_list = [a.squeeze().tolist() for a in pred_list]

    # print('True')
    # print(true_list)
    # print('Pred')
    # print(pred_list)
    
    true_list = np.array(true_list)
    pred_lsit = np.array(pred_list)
    # print(np.shape(true_list), np.shape(pred_list))
    
    # print(true_list == pred_list)
    # print(true_list-pred_list)
    
    correct = np.sum(true_list == pred_list)
    total = np.shape(true_list)
    print('Validation Accuracy: ', (correct/total)*100)
    
    cm = confusion_matrix(true_list, pred_list)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
        display_labels=['No defect', 'Defect'])
    disp.plot()
    
def testExp(dataloader):
    true_list = []
    pred_list = []
    Net.eval()
    Net.cpu() #cuda()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)
            # print('shape labels', np.shape(labels))

            if torch.cuda.is_available():
                images = images.cpu() # cuda()
                labels = labels.cpu() #cuda()
                
            for i in range(len(labels.numpy())):
                true_list.append(labels.numpy()[i]) 
            # true_list.append(labels.numpy())
            output = Net(images)
            output = torch.sigmoid(output)
            pred_tag = torch.round(output)
            [pred_list.append(pred_tag[i]) for i in range(len(pred_tag.squeeze().cpu().numpy()))]
 
    pred_list = [a.squeeze().tolist() for a in pred_list]
    
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    
    correct = np.sum(true_list == pred_list)
    total = np.shape(true_list)
    accuracy = (correct/total)*100
    
    precision, recall, f_score, support = precision_recall_fscore_support(true_list, pred_list)
    
    print('')
    print('Test Exp')
    dispResults(true_list, pred_list, precision, recall, f_score, accuracy)
    
    return precision[1], recall[1], f_score[1], accuracy

def dispResults(true_list, pred_list, precision, recall, f_score, accuracy):
    print('Confusion matrix')
    cm = confusion_matrix(true_list, pred_list)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
        display_labels=['No defect', 'Defect'])
    disp.plot()
    
    print("")
    print('Accuracy: ', accuracy)
    print('Precision ', precision[1])
    print('Recall ', recall[1])
    print('F score ', f_score[1])    
    

# y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

##############################################
# Execute the Final Training Function
##############################################
precisions = []
recalls = []
f_scores = []
accuracies = []

precisions_test = []
recalls_test = []
f_scores_test = []
accuracies_test = []

global_start_time = time.time()

for i in range(100):
          
    input_shape = (hp.channels, hp.img_size, hp.img_size)

    Net = Network(input_shape)
   
    print('')
    # print(Net)

    if i == 0:
        print(summary(Net.float(), input_size=(hp.batch_size,1,64,64)))

    print('Model iter', i)

    Net = Net.double()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(Net.parameters(), lr=hp.lr, momentum=hp.momentum)
 
    if cuda:
        Net = Net.cuda()
        criterion = criterion.cuda()
    
    train(
        train_dataloader=experimental_dataloader_train,
        n_epochs=hp.n_epochs,
        criterion=criterion,
        optimizer = optimizer,
        Tensor=Tensor,
        sample_interval=hp.sample_interval,
    )
        
    # test()
    precision, recall, f_score, accuracy = testExp(experimental_dataloader_test)
    # precision, recall, f_score, accuracy = testExp(experimental_dataloader_val)
    
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)
    accuracies.append(accuracy)
    
    # precision, recall, f_score, accuracy = testExp(experimental_dataloader_test)
    # precisions_test.append(precision)
    # recalls_test.append(recall)
    # f_scores_test.append(f_score)
    # accuracies_test.append(accuracy)

z = np.stack((accuracies,accuracies_test)).squeeze()
print(z.T[np.argsort(z.T[:, 0])])

plt.figure()
plt.plot(np.arange(0, len(accuracies)), sorted(accuracies))
plt.title('Accuracy')
plt.xlabel('Individual Models')
plt.ylabel('Accuracy')

plt.figure()
plt.hist(accuracies, bins = 5)
plt.title('Accuracy')


plt.figure()
plt.plot(precisions)
plt.title('Precision')

plt.figure()
plt.plot(f_scores)
plt.title('F1 Score')


plt.figure()
plt.plot(recalls)
plt.title('Recall')

print('Total time ', datetime.timedelta(seconds = time.time()-global_start_time))
