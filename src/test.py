#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import math
import argparse
from operator import attrgetter
from bisect import bisect_left

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

import models
from data import load_data
from optim import CentroidSGD
from quantization import PQ
from utils.training import finetune_centroids, evaluate
from utils.watcher import ActivationWatcher
from utils.dynamic_sampling import dynamic_sampling
from utils.statistics import compute_size
from utils.utils import centroids_from_weights, weight_from_centroids

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import gc
from models import resnet
from models1.real_nvp.real_nvp import real_nvp_model
from models1.real_nvp import real_nvp_loss
import util

from models1.real_nvp import real_nvp_loss

torch.cuda.empty_cache()


def main():
    torch.cuda.empty_cache()
    
    student = real_nvp_model(pretrained=True) # resnet.resnet18_1(pretrained=True).cuda()
    student.eval()
    cudnn.benchmark = True


    criterion=real_nvp_loss.RealNVPLoss().cuda()
    transform_train = transforms.Compose([ transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)



    transform_test = transforms.Compose([ transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True,     transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)


    # parameters for the centroids optimizer
    opt_centroids_params_all = []

    # book-keeping for compression statistics (in MB)
    size_uncompressed = compute_size(student) #44.591949462890625 mb 
    size_index = 0
    size_centroids = 0
    size_other = size_uncompressed

    teacher = real_nvp_model(pretrained=True)
    teacher.eval()



    layer='module.flows.in_couplings.0.st_net.blocks.0.in_conv.conv'


    n_iter_activations = math.ceil(1024 /32)
    watcher = ActivationWatcher(student, layer=layer)
    in_activations_current = watcher.watch(trainloader, criterion, n_iter_activations)
    in_activations_current = in_activations_current[layer]

    M = attrgetter(layer + '.weight.data')(student).detach()
    sizes = M.size()
    is_conv = len(sizes) == 4


    padding = attrgetter(layer)(student).padding if is_conv else 0
    stride = attrgetter(layer)(student).stride if is_conv else 1
    groups = attrgetter(layer)(student).groups if is_conv else 1

    if is_conv:
        
        out_features, in_features, k, _ = sizes
        block_size = 9 if k > 1 else 4
        n_centroids = 128 if k > 1 else 128
        n_blocks = in_features * k * k // block_size
    else:
        k = 1
        out_features, in_features = sizes
        block_size = 4
        n_centroids = 256
        n_blocks = in_features // block_size
    
    
    powers = 2 ** np.arange(0, 16, 1)
    n_vectors = np.prod(sizes) / block_size  #4096.0
    idx_power = bisect_left(powers, n_vectors / 4)
    n_centroids = min(n_centroids, powers[idx_power - 1]) #128

    # compression rations
    bits_per_weight = np.log2(n_centroids) / block_size #0.7778
    # number of bits per weight
    size_index_layer = bits_per_weight * M.numel() / 8 / 1024 / 1024
    size_index += size_index_layer #0.00341796875
        # centroids stored in float16
    size_centroids_layer = n_centroids * block_size * 2 / 1024 / 1024
    size_centroids += size_centroids_layer
    # size of non-compressed layers, e.g. BatchNorms or first 7x7 convolution
    size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
    size_other -= size_uncompressed_layer
    n_samples=1000

    # quantizer
    quantizer = PQ(in_activations_current, M, n_activations=1024,
                       n_samples=n_samples, eps=1e-8, n_centroids=n_centroids,
                       n_iter=100, n_blocks=n_blocks, k=k,
                       stride=stride, padding=padding, groups=groups)




    # quantize layer
    quantizer.encode()
    M_hat = quantizer.decode()
    attrgetter(layer + '.weight')(student).data = M_hat


    parameters = []
    parameters=[attrgetter(layer + '.weight.data')(student).detach()]
    assignments = quantizer.assignments
    centroids_params = {'params': parameters,
                            'assignments': assignments,
                            'kernel_size': k,
                            'n_centroids': n_centroids,
                            'n_blocks': n_blocks}




    opt_centroids_params = [centroids_params]
    optimizer_centroids = CentroidSGD(opt_centroids_params,         lr=0.01,momentum=0.9,weight_decay=0.0001)
    finetune_centroids(trainloader, student.eval(), teacher, criterion, optimizer_centroids, n_iter=10)





    bpd = evaluate(testloader, student, criterion)
    print('bits per dim:{:.4f} '.format(bpd))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids, step_size=1, gamma=0.1)


    # saving
    M_hat = attrgetter(layer + '.weight')(student).data
    centroids = centroids_from_weights(M_hat, assignments, n_centroids, n_blocks)
    quantizer.centroids = centroids
    quantizer.save('', layer)


if __name__ == '__main__':
    main()