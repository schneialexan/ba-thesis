import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from model import FlowTransformer
from dataset import TrainDataset, TestDataset
from utils import fullImageOut

import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#best_config = pd.read_csv('vit_study.csv')

batch_size = 32
num_epochs = 150

data = TrainDataset()
test_data = TestDataset(data)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print('Data loaded: train:', len(train_loader), 'test:', len(test_loader))

model = FlowTransformer(input_size=(3, 128, 128), output_size=(3, 128, 128)).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

                    outputs = model(images, images)
                    vali_loss += criterion(outputs, labels).item()
                    
                    if i == 0 and epoch % 5 == 0:
                        input_image = images[0].cpu().numpy()
                        target_image = labels[0].cpu().numpy()
                        output_image = outputs[0].cpu().numpy()
                        fullImageOut(f'images/output_{epoch}', input_image, target_image, output_image)
                        plt.close('all')
                    pbar_inner.set_postfix(vali_loss=vali_loss/len(test_loader))
        
        pbar.set_postfix(loss=loss_total/len(train_loader), vali_loss=vali_loss/len(test_loader))
        pbar.update()

print('Finished Training')

torch.save(model.state_dict(), 'final_vit_model.pth')

# plot and save loss
fig = plt.figure()
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss.png')

with open('losses.txt', 'w') as f:
    for loss in losses:
        f.write(f'{loss}\n')                    