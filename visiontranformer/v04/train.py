import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from model import FlowTransformer
from dataset import TrainDataset, TestDataset
from utils import fullImageOut, normalize_image

import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_config = pd.read_csv('vit_study.csv')
best_config = best_config.loc[best_config['value'].idxmin()]

num_heads = best_config['params_num_heads']
hidden_dim = best_config['params_hidden_dim']
num_layers = best_config['params_num_layers']
learning_rate = best_config['params_learning_rate']
betas = (best_config['params_beta1'], best_config['params_beta2'])
weight_decay = best_config['params_weight_decay']

#print(f'Best config: {best_config}')

batch_size = 32
num_epochs = 1000

data = TrainDataset()
test_data = TestDataset(data)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"Train Dataset length: {len(data)} | Batch size: {batch_size} | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")
print(f"Test Dataset length: {len(test_data)} | Batch size: {batch_size} | Number of batches: {len(test_loader)} | Sample shape: {test_loader.dataset[0][0].shape}")

input_size = data[0][0].shape
output_size = data[0][1].shape
model = FlowTransformer(input_size=input_size, 
                        output_size=output_size,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

total_step = len(train_loader)
loss_total = 0
losses = []

with tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        with tqdm(train_loader, leave=False) as pbar_inner:
            for i, (images, labels) in enumerate(pbar_inner):
                images = images.float().to(device)
                labels = labels.float().to(device)
                
                images = normalize_image(images)
                labels = normalize_image(labels)

                # Forward pass
                outputs = model(images, images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_total += loss.item()
                losses.append(loss.item())

                pbar_inner.set_postfix(loss=loss.item())
                pbar_inner.update()
        model.eval()
        with torch.no_grad():
            vali_loss = 0
            for i, (images, labels) in enumerate(test_loader):
                images = images.float().to(device)
                labels = labels.float().to(device)
                
                images = normalize_image(images)
                labels = normalize_image(labels)

                outputs = model(images, images)
                vali_loss += criterion(outputs, labels).item()
                
                if i == 0 and epoch % 5 == 0:
                    input_image = images[0].cpu().numpy()
                    target_image = labels[0].cpu().numpy()
                    output_image = outputs[0].cpu().numpy()
                    fullImageOut(f'images/output_{epoch}.png', input_image, target_image, output_image)
                    plt.close('all')
        
        pbar.set_postfix(loss=loss_total/total_step, vali_loss=vali_loss/len(test_loader))
        pbar.update()

print('Finished Training')

torch.save(model.state_dict(), 'final_unet_model.pth')

# plot and save loss
fig = plt.figure()
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss.png')

with open('losses.txt', 'w') as f:
    for loss in losses:
        f.write(f'{loss}\n')                    