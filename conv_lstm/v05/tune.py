'''
Hyperparameter tuning for the ConvLSTM model with Optuna
'''
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import ConvLSTM2DDataset, ValiDataset
from model import ConvLSTM
from utils import normalize_image

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

# Dataset and DataLoader
dataset = ConvLSTM2DDataset()
test_dataset = ValiDataset(dataset)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Train Dataset length: {len(dataset)} | Batch size: 32 | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")

# Model
def objective(trial):
    
    hidden_dim_choices = [32, 64, 128, 256]
    kernel_choices = [3, 5, 7]
    hidden_dim_size = trial.suggest_int('hidden_dim_size', 1, 5)
    config = {
        'hidden_dim': [trial.suggest_categorical(f'hidden_dim_{i}', hidden_dim_choices) for i in range(hidden_dim_size)],
        'kernel_size': trial.suggest_categorical('kernel_size', kernel_choices),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'beta1': trial.suggest_float('beta1', 0.9, 0.999, log=True),
        'beta2': trial.suggest_float('beta2', 0.9, 0.999, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    }
    
    model = ConvLSTM(input_dim=3, 
                    hidden_dim=config["hidden_dim"],
                    input_seq=1,
                    kernel_size=[(config["kernel_size"], config["kernel_size"])]*hidden_dim_size,
                    num_layers=hidden_dim_size).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=(config["beta1"], config["beta2"]))
    
    permutation = [0, 4, 3, 2, 1]

    # Training loop
    total_step = len(train_loader)
    loss_total = 0
    losses = []

    for epoch in range(5):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)       
            
            inputs = normalize_image(inputs)
            targets = normalize_image(targets)
            
            # Forward pass
            outputs = model(inputs).permute(permutation)
            loss_l1 = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss_l1.backward()
            optimizer.step()

            loss_total += loss_l1.item()

        losses.append(loss_total / total_step)
        loss_total = 0

    return losses[-1]

study_name = "conv_lstm_study"
storage = RDBStorage(url="sqlite:///optuna.db")
study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=4)

# Save the study to a file
study.trials_dataframe().to_csv(f"{study_name}.csv")
