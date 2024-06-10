import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import os
import tqdm
import matplotlib.pyplot as plt

from dataset import generate_input, generate_output
from model import FlowTransformer
from utils import normalize_image, fullImageOut

best_config = pd.read_csv('vit_study.csv')
best_config = best_config.loc[best_config['value'].idxmin()]

num_heads = best_config['params_num_heads']
hidden_dim = best_config['params_hidden_dim']
num_layers = best_config['params_num_layers']
learning_rate = best_config['params_learning_rate']
betas = (best_config['params_beta1'], best_config['params_beta2'])
weight_decay = best_config['params_weight_decay']

model_path = 'final_vit_model.pth'
input_size = (3, 128, 128)
output_size = (3, 128, 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FlowTransformer(input_size=input_size, 
                        output_size=output_size,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers).to(device)

model.load_state_dict(torch.load(model_path))

criterion_l1 = nn.L1Loss()
criteria_mse = nn.MSELoss()

Res = np.arange(150., 4000., 100.)
tss = np.arange(0.5, 9.95, 0.05)

losses_l1 = {}
losses_mse = {}

with tqdm.tqdm(total=len(Res)) as pbar:
    for Re in Res:
        for ts in tss:
            input = normalize_image(generate_input(Re, ts).reshape(1, 3, 128, 128))
            target = normalize_image(generate_output(Re).reshape(1, 3, 128, 128))
            
            input = torch.tensor(input, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)

            # Forward pass
            outputs = model(input)
            loss_l1 = criterion_l1(outputs, target)
            loss_mse = criteria_mse(outputs, target)            
            
            losses_l1[(Re, ts)] = loss_l1.item()
            losses_mse[(Re, ts)] = loss_mse.item()
        
        pbar.update(1)
        pbar.set_description(f"Re: {Re:.1f}")

# average the losses per RE and TS
losses_l1_re_avg = {}
losses_l1_ts_avg = {}
losses_mse_re_avg = {}
losses_mse_ts_avg = {}

for Re in Res:
    losses_l1_re_avg[Re] = np.mean([losses_l1[(Re, ts)] for ts in tss])
    losses_mse_re_avg[Re] = np.mean([losses_mse[(Re, ts)] for ts in tss])

for ts in tss:
    losses_l1_ts_avg[ts] = np.mean([losses_l1[(Re, ts)] for Re in Res])
    losses_mse_ts_avg[ts] = np.mean([losses_mse[(Re, ts)] for Re in Res])
    
# plot the losses
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# plot the RE averaged losses
axes[0].plot(Res, [losses_l1_re_avg[Re] for Re in Res], label='L1 Loss')
axes[0].plot(Res, [losses_mse_re_avg[Re] for Re in Res], label='MSE Loss')
axes[0].set_xlabel('Re')
axes[0].set_ylabel('Loss')
axes[0].legend()

# plot the TS averaged losses
axes[1].plot(tss, [losses_l1_ts_avg[ts] for ts in tss], label='L1 Loss')
axes[1].plot(tss, [losses_mse_ts_avg[ts] for ts in tss], label='MSE Loss')
axes[1].set_xlabel('ts')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('test_losses.png')
plt.show()

# convert to pandas dataframe
import pandas as pd
df = pd.DataFrame({
    'Re': [Re for Re in Res for ts in tss],
    'ts': [ts for Re in Res for ts in tss],
    'L1 Loss': [losses_l1[(Re, ts)] for Re in Res for ts in tss],
    'MSE Loss': [losses_mse[(Re, ts)] for Re in Res for ts in tss]
})

df.to_csv('test_losses.csv', index=False)

worst_Re, worst_ts = max(losses_l1, key=losses_l1.get)
print(f"Worst L1 Loss: Re={worst_Re}, ts={worst_ts}, L1 Loss={losses_l1[(worst_Re, worst_ts)]}")
input = normalize_image(generate_input(worst_Re, worst_ts).reshape(1, 3, 128, 128))
target = normalize_image(generate_output(worst_Re).reshape(1, 3, 128, 128))

input = torch.tensor(input, dtype=torch.float32).to(device)
target = torch.tensor(target, dtype=torch.float32).to(device)

outputs = model(input)

fullImageOut('worst_l1_loss', input[0].cpu().detach().numpy(), target[0].cpu().detach().numpy(), outputs[0].cpu().detach().numpy())