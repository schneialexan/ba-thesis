import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import os
import tqdm
import matplotlib.pyplot as plt

from dataset import generate_input, generate_output
from model import UNet
from utils import normalize_image

model_path = 'final_unet_model.pth'
expo = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(channelExponent=3, dropout=0.).to(device)
model.load_state_dict(torch.load(model_path))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, betas=(0.5, 0.999), weight_decay=0.0)


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
        # average the losses of one Reynolds number
        losses[Re] = np.mean(loss_re)
        
        pbar.update(1)
        pbar.set_description(f"Re: {Re:.1f} | Loss: {losses[Re]:.4f}")

fig, ax = plt.subplots()
plt.title('Losses for unseen Reynolds numbers')
ax.plot(Res, [losses[Re] for Re in Res])
ax.set_xlabel('Reynolds number')
ax.set_ylabel('Averaged loss')
plt.savefig('losses.png')