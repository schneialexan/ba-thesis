import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from imagetrans import ImageTransformer, train, evaluate
from dataset import TrainDataset, ValiDataset

batch_size = 32

data = TrainDataset()
dataValidation = ValiDataset(data)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
testLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True) 


N_EPOCHS = 150

imagesize = trainLoader.dataset[0][0].shape[1]
patchsize = 4
num_classes = 10
channels = trainLoader.dataset[0][0].shape[0]
dim = 64
depth = 2
heads = 8
mlp_dim = 128

model = ImageTransformer(image_size=imagesize, 
                         patch_size=patchsize, 
                         num_classes=num_classes, 
                         channels=channels, 
                         dim=dim, 
                         depth=depth, 
                         heads=heads, 
                         mlp_dim=mlp_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, trainLoader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, testLoader, test_loss_history)

print('Execution time')

PATH = ".\ViTnet_Cifar10_4x4_aug_1.pt" # Use your own path
torch.save(model.state_dict(), PATH)