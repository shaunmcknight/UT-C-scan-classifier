# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:19:42 2022

@author: Shaun McKnight
"""

import torch
import numpy as np


from utils import *
from classifier import *
from train import *

def sampleDataImage(dataloader):
    test_data = next(iter(dataloader))
    plt.figure()
    plt.imshow(test_data[0][0].squeeze(), cmap='gray')
    # plt.clim(0,1)
    plt.colorbar()

def experimental(HP, transforms, exp_path, iteration):

    # root_path = r"C:\GIT\basic_classifier\Experimental\HPO_chris\experimental"
    root_path = exp_path # "C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/experimental"
    root_train = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\experimental_train"

    experimental_dataloader = DataLoader(
        ExperimentalDataset(root_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    split_train, split_test = round(len(experimental_dataloader.dataset)*0.8), round(
        len(experimental_dataloader.dataset))-round(len(experimental_dataloader.dataset)*0.8)
    
    train_exp, test_exp = torch.utils.data.random_split(
        experimental_dataloader.dataset, (split_train, split_test))
    
    train_dataloader = DataLoader(
        train_exp,
        batch_size=hp.batch_size,
        shuffle=True,
    )
    
    test_dataloader = DataLoader(
        test_exp,
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    
    # train_dataloader = DataLoader(
    #     ExperimentalDataset(root_train, transforms_=transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=True,
    # )
    
    # split_train, split_test = round(len(train_dataloader.dataset)*1), round(
    #     len(train_dataloader.dataset))-round(len(train_dataloader.dataset)*1)
    
    # train_exp, test_exp = torch.utils.data.random_split(
    #     train_dataloader.dataset, (split_train, split_test))
    
    # train_dataloader = DataLoader(
    #     train_exp,
    #     batch_size=hp.batch_size,
    #     shuffle=True,
    # )
    
    # print('Train len: ', len(train_dataloader.dataset))
    
    # experimental_dataloader = DataLoader(
    #     ExperimentalDataset(root_path, transforms_=transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    # )
    
    # train_dataloader = train_dataloader
    # test_dataloader = experimental_dataloader
    
    sampleDataImage(experimental_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
         validation_dataloader=None, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm
    
def GAN(HP, transforms, exp_path, iteration):

    root_path = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\synthetic\GAN"

    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    # valid_dataloader = DataLoader(
    #     ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    #     # num_workers=1,
    # )
    
    test_dataloader = DataLoader(
        ExperimentalDataset(exp_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )

    sampleDataImage(train_dataloader)
    # sampleDataImage(valid_dataloader)
    sampleDataImage(test_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
         validation_dataloader=None, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm

def CIVA(HP, transforms, exp_path, iteration):

    root_path = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\classifier\simple\civa"
    
    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    valid_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
        # num_workers=1,
    )
    
    test_dataloader = DataLoader(
        ExperimentalDataset(exp_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    

    sampleDataImage(train_dataloader)
    sampleDataImage(valid_dataloader)
    sampleDataImage(test_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
         validation_dataloader=valid_dataloader, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm

def imageNoiseSim(HP, transforms, exp_path, iteration):

    root_path = r"C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/synthetic/image_noise/noised_image_level"
    
    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    
    # Not doing validation due to lack only 79 datapoints once noised up
    # valid_dataloader = DataLoader(
    #     ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    #     # num_workers=1,
    # )
    
    test_dataloader = DataLoader(
        ExperimentalDataset(exp_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )

    sampleDataImage(train_dataloader)
    # sampleDataImage(valid_dataloader)
    sampleDataImage(test_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
          validation_dataloader=None, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm


def scanNoiseSim(HP, transforms, exp_path, iteration):

    root_path = r"C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/synthetic/image_noise/noised_scan_level"
    
    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    
    # Not doing validation due to lack only 79 datapoints once noised up
    # valid_dataloader = DataLoader(
    #     ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    #     # num_workers=1,
    # )
    
    test_dataloader = DataLoader(
        ExperimentalDataset(exp_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    sampleDataImage(train_dataloader)
    # sampleDataImage(valid_dataloader)
    sampleDataImage(test_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
          validation_dataloader=None, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm

def imageNoiseReal(HP, transforms, exp_path, iteration):

    root_path = r"C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/synthetic/image_noise/real_noise"
    
    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    
    # Not doing validation due to lack only 79 datapoints once noised up
    # valid_dataloader = DataLoader(
    #     ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = transforms_),
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    #     # num_workers=1,
    # )
    
    test_dataloader = DataLoader(
        ExperimentalDataset(exp_path, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )

    sampleDataImage(train_dataloader)
    # sampleDataImage(valid_dataloader)
    sampleDataImage(test_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
          validation_dataloader=None, 
         test_dataloader=test_dataloader, 
         cuda_device=None,
         iteration = iteration)

    return accuracy, precision, recall, f_score, cm

# HP = {'n_epochs': 250,
#       'batch_size': 128, #64
#       'lr': 0.03, #0.03,
#       'momentum': 0.9,
#       'early_stop': 2,
#       'conv_layers': 4,
#       'out_channel_ratio': 4,
#       'FC_layers': 3
#       }


#optimisation of experimental data
HP = {'n_epochs': 264,
      'batch_size': 4, #64
      'lr': 0.013870869810956584, #0.03,
      'momentum': 0.175764011181887,
      'early_stop': 1,
      'conv_layers': 3,
      'out_channel_ratio': 3,
      'FC_layers': 1
      }

#optimised to C scan sim noise
#{'FC_layers': 1, 'batch_size': 4, 'conv_layers': 4, 'early_stop': 0, 'lr': 0.0020317484998939317, 'momentum': 0.9733433853905421, 'n_epochs': 240, 'out_channel_ratio': 3}
# HP = {'n_epochs': 240,
#       'batch_size': 4, #64
#       'lr': 0.0020317484998939317, #0.03,
#       'momentum': 0.9733433853905421,
#       'early_stop': 0,
#       'conv_layers': 4,
#       'out_channel_ratio': 3,
#       'FC_layers': 1
#       }

hp = Hyperparameters(
    epoch=0,
    n_epochs=HP['n_epochs'],
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=2**HP['batch_size'],
    lr=HP['lr'],
    momentum=HP['momentum'],
    img_size=64,
    channels=1,
    early_stop=HP['early_stop']
)


transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
]

exp_path = "C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/experimental"

accuracies = []
precisions = []
recalls = []
f_scores = []
confusion_matrixes = []
trps = []
fprs = []

for i in range(1):

    print('Model iteration ~ ', i)
    
    """Experimental"""
    
    # accuracy, precision, recall, f_score, cm = experimental(HP, transforms=transforms_, exp_path=exp_path, iteration = i)
    
    
    """GAN"""
    
    # accuracy, precision, recall, f_score, cm = GAN(HP, transforms=transforms_, exp_path=exp_path, iteration = i)
    
    
    """CIVA"""
    
    # accuracy, precision, recall, f_score, cm = CIVA(HP, transforms=transforms_, exp_path=exp_path, iteration = i)


    """C scan level noise distribution"""
    
    # accuracy, precision, recall, f_score, cm = imageNoiseSim(HP, transforms=transforms_, exp_path=exp_path, iteration = i)
    

    """C scan level real noise"""
    
    # accuracy, precision, recall, f_score, cm = imageNoiseReal(HP, transforms=transforms_, exp_path=exp_path, iteration = i)   
        
        
    """B scan level noise distribution"""
    
    accuracy, precision, recall, f_score, cm = scanNoiseSim(HP, transforms=transforms_, exp_path=exp_path, iteration = i)


    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)
    confusion_matrixes.append(cm)
    

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Mean results ~")
print('~~~~~~~~~~~~~~~~~')
print("")
print('Accuracy ~ mu {}. std {}. '.format(np.mean(accuracies),np.std(accuracies)))
print('Precision ~ mu {}. std {}. '.format(np.mean(precisions),np.std(precisions)))
print('Recall ~ mu {}. std {}. '.format(np.mean(recalls),np.std(recalls)))
print('F score ~ mu {}. std {}. '.format(np.mean(f_scores),np.std(f_scores)))

print('Confusion matrix')
cm = np.array(confusion_matrixes)
cm = np.mean(cm, axis = 0)
print(cm)

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Max results ~")
print('~~~~~~~~~~~~~~~~~')
print('')
print('Accuracy ~ Max {}. '.format(np.amax(accuracies)))
print('Precision ~ Max {}. '.format(np.amax(precisions)))
print('Recall ~ Max {}. '.format(np.amax(recalls)))
print('F score ~ Max {}. '.format(np.amax(f_scores)))



"""

TO DO:
    add in training lists for 100 iterations to get std and averages
    re-train GAN
    
"""
