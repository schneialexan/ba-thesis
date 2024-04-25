##################
#
# Bachelor Thesis: Deep learning based lid-driven cavity flow simulation with u-net CNN
# @author: Alexandru Schneider
# @supervisor: R.P. Mundani
# Computational and Data Science
# FHGR
#
##################

import os, sys, random, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from UNet import UNet, weights_init
import dataset
import utils
import matplotlib.pyplot as plt

######## Settings ########

# number of training iterations
iterations = 500000   # 100k original
# batch size --> how many images are processed at once
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop=None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

##########################

dropout    = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad     = ""      # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

######## Data ########
print("\nLoading data...")
data = dataset.TrainDataset(prop, shuffle=1)
dataValidation = dataset.ValiDataset(data)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True) 
print("\nTraining batches: {}".format(len(trainLoader)))
print("Validation batches: {}".format(len(valiLoader)))

######## Setup ########
print("\nSetting up model...")
epochs = int(iterations / len(trainLoader) + 0.5)
model = UNet(channelExponent=expo, dropout=dropout)
# save model as onnx to show with netron
dummy_input = torch.randn(batch_size, 3, 128, 128)
torch.onnx.export(model, dummy_input, "model_dummy_data.onnx")
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized model with {} parameters\n".format(params))

# initialize weights and load model if necessary
model.apply(weights_init)
if len(doLoad)>0:
    model.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
    
# loss function
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs  = Variable(torch.FloatTensor(batch_size, 3, 128, 128))

curr_L1 = 0.
# train
with tqdm.tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        pbar.set_description(f'Epoch: {epoch}, L1: {curr_L1:.5f} | Current Epoch: {epoch+1}')
        
        # training
        model.train()
        L1_accum = 0
        for i, train in enumerate(trainLoader, 0):
            inputs_cpu, targets_cpu = train
            inputs.data.copy_(inputs_cpu.float())
            targets.data.copy_(targets_cpu.float())

            # compute LR decay
            if decayLr:
                currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
                if currLr < lrG:
                    for g in optimizer.param_groups:
                        g['lr'] = currLr

            model.zero_grad()
            gen_out = model(inputs)

            lossL1 = criterion(gen_out, targets)
            lossL1.backward()

            optimizer.step()

            lossL1viz = lossL1.item()
            L1_accum += lossL1viz

            if i==len(trainLoader)-1:
                utils.writeLog("log.txt", epoch, i, lossL1viz)
                curr_L1 = lossL1viz


        # validation
        model.eval()
        L1val_accum = 0.0
        for i, validata in enumerate(valiLoader, 0):
            inputs_cpu, targets_cpu = validata
            inputs.data.copy_(inputs_cpu.float())
            targets.data.copy_(targets_cpu.float())

            outputs = model(inputs)
            outputs_cpu = outputs.data.cpu().numpy()

            lossL1 = criterion(outputs, targets)
            L1val_accum += lossL1.item()

            if i==0 and epoch % 5 == 0:
                outputs_denormalized = utils.normalize(outputs_cpu[0])
                targets_denormalized = utils.normalize(targets_cpu.cpu().numpy()[0])
                utils.makeDirs(["results_train"])
                utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_denormalized, targets_denormalized, saveTargets=True)
                plt.close('all')
        pbar.update(1)
    

torch.save(model.state_dict(), "model")