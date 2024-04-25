'''
1. Test the U-Net model after training.
2. Create plots of the loss (L1) over the epochs.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

#########################################
# make plot for loss over epochs
#########################################
def plot_loss(logfile):
    data = pd.read_csv(logfile, header=None)
    data.columns = ['epoch', 'batch', 'loss']
    data['epoch'] = data['epoch'].str.extract(r'(\d+)').astype(int)
    data['batch'] = data['batch'].str.extract(r'(\d+)').astype(int)
    data['loss'] = data['loss'].str.extract(r'(\d+\.\d+)').astype(float)

    plt.plot(data["epoch"], data["loss"], label="L1 Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.xticks(np.arange(0, data["epoch"].max(), 50.))
    plt.savefig("loss.png")
    #plt.show()
    
plot_loss("log.txt")


#########################################
# test the model with a sample input from an unknown Reynolds number
#########################################
from UNet import UNet
import utils

import os, sys, random, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from UNet import UNet, weights_init
import dataset
import utils
import matplotlib.pyplot as plt

######## Settings ########
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
doLoad     = "model"      # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

######## Setup ########
print("\nSetting up model...")
model = UNet(channelExponent=expo, dropout=dropout)
if len(doLoad)>0:
    model.load_state_dict(torch.load(doLoad))
    print("Loaded model: "+doLoad + " successfully!")
    
# loss function
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

# get a sample input
Re = 750.
ts = 0.5

Res = np.arange(150., 4000., 100.)
tss = np.arange(0.5, 10.0, 0.05)

losses = []

for Re in tqdm.tqdm(Res):
    ts = random.choice(tss)
    ts = f'{ts:.2f}'
    tqdm.tqdm.write(f"Re: {Re}, ts: {ts}")
    input_mask = dataset.gen_mask(Re, 128)
    input_u = dataset.open_image(f"../../data/{Re:.1f}/u_{ts:.2f}.png")
    input_v = dataset.open_image(f"../../data/{Re:.1f}/v_{ts:.2f}.png")
    
    input = np.stack([input_mask, input_u[0], input_v[0]], axis=0)
    input = torch.from_numpy(input).float().unsqueeze(0)
    
    target_p = dataset.open_image(f"../../data/{Re:.1f}/p_ss.png")
    target_u = dataset.open_image(f"../../data/{Re:.1f}/u_ss.png")
    target_v = dataset.open_image(f"../../data/{Re:.1f}/v_ss.png")
    
    target = np.stack([target_p[0], target_u[0], target_v[0]], axis=0)
    target = torch.from_numpy(target).float().unsqueeze(0)
    
    # predict the output
    output = model(input)
    l1_loss = criterion(output, target)
    losses.append(l1_loss.item())
    
    # plot the input and output
    input_norm = utils.normalize(input[0].numpy())
    target_norm = utils.normalize(target[0].numpy())
    output_norm = utils.normalize(output[0].detach().numpy())
    
    utils.makeDirs(["results_test"])
    utils.imageOut("results_test/test_{}_{}".format(Re, ts), output_norm, target_norm, saveTargets=True)
    utils.fullImageOut("results_test/test_{}_{}".format(Re, ts), input_norm, output_norm, target_norm, saveTargets=True)
    plt.close()


print(f'Average Loss: {np.mean(losses)}')

fig, ax = plt.subplots()
ax.plot(Res, losses)
ax.set(xlabel='Re', 
       ylabel='L1 Loss',
       title='L1 Loss over Reynolds Number')
ax.grid()
plt.savefig("loss_re.png")
plt.show()
