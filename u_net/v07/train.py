import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pynvml
import os

pynvml.nvmlInit()

from model import UNet
from dataset import TrainDataset, TestDataset
from utils import fullImageOut, normalize_image

import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_epochs = 100
patience = 10  # Number of epochs to wait for improvement before stopping
tolerance = 0.01

data = TrainDataset()
test_data = TestDataset(data)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"Train Dataset length: {len(data)} | Batch size: {batch_size} | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")
print(f"Test Dataset length: {len(test_data)} | Batch size: {batch_size} | Number of batches: {len(test_loader)} | Sample shape: {test_loader.dataset[0][0].shape}")

model = UNet(channelExponent=3, dropout=0.).to(device)

expo = 3
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, betas=(0.5, 0.999), weight_decay=0.0)

total_step = len(train_loader)
train_losses = []
val_losses = []
best_vali_loss = float('inf')

# Resource monitoring
cpu_usage = []
ram_usage_gb = []
gpu_mem_usage = []
gpu_percent_usage = []
times = []

def get_ressource_monitoring():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    process_id = os.getpid()
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    # Find the memory usage for the current process
    memory_usage = 0
    for process in processes:
        if process.pid == process_id:
            memory_usage = process.usedGpuMemory
            break
    gpu_mem = memory_usage / 1024 / 1024  / 1024  # Convert to GB
    
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu = utilization.gpu
    
    process = psutil.Process(process_id)
    
    cpu = process.cpu_percent()
    ram = process.memory_info().rss / (1024 ** 3)  # Convert to GB

    # return cpu_usage, ram, gpu_usage, gpu_mem
    return cpu, ram, gpu, gpu_mem

start_time = time.time()

with tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss_epoch = 0
        with tqdm(train_loader, leave=False) as pbar_inner:
            for i, (images, labels) in enumerate(pbar_inner):
                images = images.float().to(device)
                labels = labels.float().to(device)
                
                images = normalize_image(images)
                labels = normalize_image(labels)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()

                pbar_inner.set_postfix(loss=loss.item())
                pbar_inner.update()

        train_loss_epoch /= total_step
        train_losses.append(train_loss_epoch)

        model.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            for i, (images, labels) in enumerate(test_loader):
                images = images.float().to(device)
                labels = labels.float().to(device)
                
                images = normalize_image(images)
                labels = normalize_image(labels)

                outputs = model(images)
                val_loss_epoch += criterion(outputs, labels).item()

                if i == 0 and epoch % 5 == 0:
                    input_image = images[0].cpu().numpy()
                    target_image = labels[0].cpu().numpy()
                    output_image = outputs[0].cpu().numpy()
                    fullImageOut(f'images/output_{epoch}.png', input_image, target_image, output_image)
                    plt.close('all')

        val_loss_epoch /= len(test_loader)
        val_losses.append(val_loss_epoch)

        pbar.set_postfix(train_loss=train_loss_epoch, val_loss=val_loss_epoch)
        pbar.update()

        # Resource monitoring
        cpu, ram, gpu_util, gpu_mem = get_ressource_monitoring()
        cpu_usage.append(cpu)
        ram_usage_gb.append(ram)
        gpu_percent_usage.append(gpu_util)
        gpu_mem_usage.append(gpu_mem)

        times.append(time.time() - epoch_start_time)

        # Early stopping logic
        if val_loss_epoch < best_vali_loss or abs(val_loss_epoch - best_vali_loss) > tolerance:
            best_vali_loss = val_loss_epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_unet_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if epoch % 10 == 0:
            # Plot and save train and validation loss
            fig, ax = plt.subplots()
            ax.plot(range(len(train_losses)), train_losses, label='Train Loss')
            ax.plot(range(len(val_losses)), val_losses, label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.savefig('train_val_loss.png')

            # Plot and save resource usage
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Usage (%)', color='tab:blue')
            ax1.plot(range(len(cpu_usage)), cpu_usage, label='CPU Usage (%)', color='tab:blue')
            ax1.plot(range(len(gpu_percent_usage)), gpu_percent_usage, label='GPU Usage (%)', linestyle='dotted', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Memory Usage (GB)', color='tab:orange')
            ax2.plot(range(len(ram_usage_gb)), ram_usage_gb, label='RAM Usage (GB)', linestyle='dashed', color='tab:orange')
            ax2.plot(range(len(gpu_mem_usage)), gpu_mem_usage, label='GPU Memory Usage (GB)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            fig.tight_layout()
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            ax1.set_ylim(0, 100)
            plt.savefig('resource_usage.png')
            plt.close('all')

total_time = time.time() - start_time
print(f'Finished Training in {total_time:.2f} seconds')

# Load the best model
model.load_state_dict(torch.load('best_unet_model.pth'))
torch.save(model.state_dict(), 'final_unet_model.pth')

# Plot and save train and validation loss
fig, ax = plt.subplots()
ax.plot(range(len(train_losses)), train_losses, label='Train Loss')
ax.plot(range(len(val_losses)), val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig('train_val_loss.png')

# Plot and save resource usage
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Usage (%)', color='tab:blue')
ax1.plot(range(len(cpu_usage)), cpu_usage, label='CPU Usage (%)', color='tab:blue')
ax1.plot(range(len(gpu_percent_usage)), gpu_percent_usage, label='GPU Usage (%)', linestyle='dotted', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Memory Usage (GB)', color='tab:orange')
ax2.plot(range(len(ram_usage_gb)), ram_usage_gb, label='RAM Usage (GB)', linestyle='dashed', color='tab:orange')
ax2.plot(range(len(gpu_mem_usage)), gpu_mem_usage, label='GPU Memory Usage (GB)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.savefig('resource_usage.png')

with open('resources.txt', 'w') as f:
    f.write(f'Total time: {total_time:.2f} seconds\n')
    f.write('CPU Usage (%):\n')
    f.write('\n'.join(map(str, cpu_usage)) + '\n')
    f.write('RAM Usage (GB):\n')
    f.write('\n'.join(map(str, ram_usage_gb)) + '\n')
    f.write('GPU Usage (%):\n')
    f.write('\n.join(map(str, gpu_percent_usage))' + '\n')
    f.write('GPU Memory Usage (GB):\n')
    f.write('\n'.join(map(str, gpu_mem_usage)) + '\n')
    f.write('Epoch times (seconds):\n')
    f.write('\n'.join(map(str, times)) + '\n')
    
with open('ressources.csv', 'w') as f:
    f.write('CPU Usage (%),RAM Usage (GB),GPU Usage (%),GPU Memory Usage (GB),Epoch times (seconds)\n')
    for cpu, ram, gpu, gpu_mem, time in zip(cpu_usage, ram_usage_gb, gpu_percent_usage, gpu_mem_usage, times):
        f.write(f'{cpu},{ram},{gpu},{gpu_mem},{time}\n')

with open('train_losses.txt', 'w') as f:
    f.write('\n'.join(map(str, train_losses)) + '\n')

with open('val_losses.txt', 'w') as f:
    f.write('\n'.join(map(str, val_losses)) + '\n')                 
    
pynvml.nvmlShutdown()