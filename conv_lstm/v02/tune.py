'''
Hyperparameter tuning for the ConvLSTM model with Optuna
'''
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import ConvLSTM2DDataset, ValiDataset
from model import ConvLSTM

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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train Dataset length: {len(dataset)} | Batch size: 32 | Number of batches: {len(train_loader)} | Sample shape: {train_loader.dataset[0][0].shape}")
print(f"Test Dataset length: {len(test_dataset)} | Batch size: 32 | Number of batches: {len(test_loader)} | Sample shape: {test_loader.dataset[0][0].shape}")

# Model
def objective(trial):
    config = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128), # suggested range: 16-128
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "num_layers": trial.suggest_int("num_layers", 1, 3), # suggested range: 1-3
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2), # suggested range: 1e-4-1e-2
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4), # suggested range: 1e-6-1e-4
        "beta1": trial.suggest_float("beta1", 0.5, 0.9), # suggested range: 0.5-0.9
        "beta2": trial.suggest_float("beta2", 0.5, 0.999), # suggested range: 0.5-0.999
        "criterion": trial.suggest_categorical("criterion", ['mse', 'l1'])
    }
    
    kernel_size = (config["kernel_size"], config["kernel_size"])

    model = ConvLSTM(3, config["hidden_dim"], 3, kernel_size, config["num_layers"], True, True, False).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss() if config["criterion"] == 'mse' else nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=(config["beta1"], config["beta2"]))
    
    permutation = [0, 3, 4, 2, 1]

    # Training loop
    total_step = len(train_loader)
    loss_total = 0
    losses = []

    for epoch in range(5):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            inputs = inputs.permute(permutation).contiguous()
            targets = targets.permute(permutation).contiguous()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        losses.append(loss_total / total_step)
        loss_total = 0

    return losses[-1]

study_name = "conv_lstm_study"
storage = RDBStorage(url="sqlite:///optuna.db")
study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=16)

# Save the study to a file
study.trials_dataframe().to_csv(f"{study_name}.csv")
