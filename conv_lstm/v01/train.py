import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import ConvLSTM2DDataset, ValiDataset
from model import ConvLSTM

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

print(f"Device: {device} | Learning rate: {learning_rate:.4f} | Num epochs: {num_epochs}")

# Dataset and DataLoader
dataset = ConvLSTM2DDataset()
test_dataset = ValiDataset(dataset)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Dataset length: {len(dataset)} | Batch size: {batch_size} | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")
print(f"Test Dataset length: {len(test_dataset)} | Batch size: {batch_size} | Number of batches: {len(test_loader)} | Sample shape: {test_loader.dataset[0][0].shape}")

# Model
config = {
            "hidden_dim": 32,
            "kernel_size": (3,3),
            "num_layers": 1
        }
model = ConvLSTM(3, config["hidden_dim"], 3, config["kernel_size"], config["num_layers"], True, True, False).to(device)
#torch.onnx.export(model, torch.rand((32, 10, 3, 128, 128)), "convlstm.onnx")

# Loss and optimizer
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
loss_total = 0
losses = []

print("Start training...")
permutation = [0, 3, 4, 2, 1]

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(tqdm(train_loader)):        
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        
        # permutation
        inputs = inputs.permute(permutation)
        targets = targets.permute(permutation)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion_mse(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_total += loss.item()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss_total/10:.4f}')
            losses.append(loss_total/10)
            loss_total = 0
    
    # Validation
    model.eval()
    with torch.no_grad():
        vali_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            inputs = inputs.permute(permutation)
            targets = targets.permute(permutation)
            
            outputs = model(inputs)
            vali_loss += criterion_mse(outputs, targets).item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {vali_loss/len(test_loader):.4f}')
    