import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TrainDataset
from UNet import UNet

import optuna
from optuna.storages import RDBStorage

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")

SEED = 42
torch.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TrainDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Train Dataset length: {len(dataset)} | Batch size: 32 | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")

def objective(trial):
    config = {
        "channelExponent": trial.suggest_int("channelExponent", 4, 8), # suggested range: 4-8
        "dropout": trial.suggest_float("dropout", 0.0, 0.5), # suggested range: 0.0-0.5
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2), # suggested range: 1e-4-1e-2
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4), # suggested range: 1e-6-1e-4
        "beta1": trial.suggest_float("beta1", 0.5, 0.9), # suggested range: 0.5-0.9
        "beta2": trial.suggest_float("beta2", 0.5, 0.999), # suggested range: 0.5-0.999
    }

    model = UNet(config["channelExponent"], config["dropout"]).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), 
                     lr=config["learning_rate"], 
                     weight_decay=config["weight_decay"], 
                     betas=(config["beta1"], config["beta2"]))

    # Training loop
    total_step = len(train_loader)
    loss_total = 0
    losses = []
    
    # Training loop
    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_total += loss.item()
        
        losses.append(loss_total / total_step)
        loss_total = 0
    
    return losses[-1]
        
            
study = optuna.create_study(direction="minimize", storage=RDBStorage("sqlite:///optuna.db"), study_name="unet", load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=4)
study.trials_dataframe().to_csv("unet_study.csv")

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
