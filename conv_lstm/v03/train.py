import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ConvLSTM2DDataset, ValiDataset
from model import ConvLSTM
from utils import fullImageOut

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")
import pandas as pd

best_config = pd.read_csv('conv_lstm_study.csv')
best_config = best_config.iloc[41]
best_config = best_config.to_dict()

image_dir = "images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
learning_rate = best_config["params_learning_rate"]
num_epochs = 1000

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
            "hidden_dim": best_config["params_hidden_dim"],
            "kernel_size": (best_config["params_kernel_size"], best_config["params_kernel_size"]),
            "num_layers": best_config["params_num_layers"],
            "learning_rate": best_config["params_learning_rate"],
            "beta1": best_config["params_beta1"],
            "beta2": best_config["params_beta2"],
            "weight_decay": best_config["params_weight_decay"],
        }

model = ConvLSTM(3, config["hidden_dim"], 3, config["kernel_size"], config["num_layers"], True, True, False).to(device)
#torch.onnx.export(model, torch.rand((32, 10, 3, 128, 128)), "convlstm.onnx")

# Loss and optimizer
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"])

# Training loop
total_step = len(train_loader)
loss_total = 0
losses = []

permutation = [0, 3, 4, 2, 1]

with tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        with tqdm(train_loader, leave=False) as pbar_inner:
            for i, (inputs, targets) in enumerate(train_loader):        
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                
                # permutation
                inputs = inputs.permute(permutation)
                targets = targets.permute(permutation)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion_l1(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_total += loss.item()
                
                if (i+1) % 10 == 0:
                    pbar_inner.set_description(f'Step [{i+1}/{total_step}], Loss: {loss_total/10:.4f}')
                    losses.append(loss_total/10)
                    loss_total = 0
                pbar_inner.update(1)
            
            # Validation
            model.eval()
            with torch.no_grad():
                vali_loss = 0
                pbar_inner.set_description(f'Validation')
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.float().to(device)
                    targets = targets.float().to(device)
                    
                    inputs = inputs.permute(permutation)
                    targets = targets.permute(permutation)
                    
                    outputs = model(inputs)
                    vali_loss += criterion_l1(outputs, targets).item()
                    
            
                    if i == 0 and epoch % 5 == 0:
                        inputs_image = inputs.permute([0, 4, 3, 1, 2])[0].cpu().numpy()[0]
                        output_image = outputs.permute([0, 4, 3, 1, 2])[0].cpu().numpy()[0]
                        target_image = targets.permute([0, 4, 3, 1, 2])[0].cpu().numpy()[0]
                        fullImageOut(os.path.join(image_dir, "output_%d" % epoch), inputs_image, output_image, target_image)
                        plt.close('all')
                    
                    pbar_inner.set_description(f'Validation Loss: {vali_loss/len(test_loader):.4f}')
                    pbar_inner.update(1)
        pbar.update(1)

print("Training finished!")

torch.save(model.state_dict(), 'model.pth')

# Plot loss
fig = plt.figure()
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss')
plt.savefig('loss.png')