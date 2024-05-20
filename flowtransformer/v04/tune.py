import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TrainDataset

import optuna
from optuna.storages import RDBStorage

from model import FlowTransformer

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


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def objective(trial):
    possible_heads = [2, 4, 8, 16]
    possible_dims = [128, 256, 384, 512, 640, 768, 896, 1024]
    config = {
        # num_heads=8, hidden_dim=512, num_layers=1
        "num_heads": trial.suggest_categorical("num_heads", possible_heads),
        "hidden_dim": trial.suggest_categorical("hidden_dim", possible_dims),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4),
        "beta1": trial.suggest_float("beta1", 0.5, 0.9),
        "beta2": trial.suggest_float("beta2", 0.5, 0.999),
    }

    model = FlowTransformer((3, 128, 128), (3, 128, 128), 
                            num_heads=config["num_heads"], 
                            hidden_dim=config["hidden_dim"], 
                            num_layers=config["num_layers"]).to(device)

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
            
            images = normalize(images)
            labels = normalize(labels)

            # Forward pass
            outputs = model(images, images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_total += loss.item()
        
        losses.append(loss_total / total_step)
        loss_total = 0
    
    return losses[-1]
        
            
study = optuna.create_study(direction="minimize", storage=RDBStorage("sqlite:///optuna.db"), study_name="FlowTransformer", load_if_exists=True)
study.optimize(objective, n_trials=1000, n_jobs=-1)
study.trials_dataframe().to_csv("vit_study.csv")

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
