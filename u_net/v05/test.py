import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from dataset import generate_input, generate_output
from model import UNet
from utils import normalize_image, fullImageOut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_config = pd.read_csv('unet_study.csv')
best_config = best_config.loc[best_config['value'].idxmin()]

learning_rate = best_config['params_learning_rate']
betas = (best_config['params_beta1'], best_config['params_beta2'])
channel_exponent = best_config['params_channelExponent']
dropout = best_config['params_dropout']
weight_decay = best_config['params_weight_decay']

model_path = 'final_unet_model.pth'
model = UNet(channelExponent=channel_exponent, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path))

# Define the loss function
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

Res = np.arange(150., 4000., 100.)
tss = np.arange(0.5, 9.95, 0.05)

losses = {}

with tqdm.tqdm(total=len(Res)) as pbar:
    for Re in Res:
        loss_re = []
        for ts in tss:
            input = normalize_image(generate_input(Re, ts).reshape(1, 3, 128, 128))
            target = normalize_image(generate_output(Re).reshape(1, 3, 128, 128))
            
            input = torch.tensor(input, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)

            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, target)
            
            loss_re.append(loss.item())
            inputs = input.cpu().detach().numpy()[0]
            targets = target.cpu().detach().numpy()[0]
            outputs = outputs.cpu().detach().numpy()[0]
        fullImageOut('testout', inputs, targets, outputs)
        # average the losses of one Reynolds number
        losses[Re] = np.mean(loss_re)
        exit()
        pbar.update(1)
        pbar.set_description(f"Re: {Re:.1f} | Loss: {losses[Re]:.4f}")

fig, ax = plt.subplots()
ax.plot(Res, [losses[Re] for Re in Res])
ax.set_xlabel('Reynolds number')
ax.set_ylabel('Loss')
plt.savefig('losses.png')