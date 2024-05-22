'''
Dataset class for the ConvLSTM 2D model

3 grayscale images are stacked together to form a single input sample
- The first image is the current problem (called mask, consisting of the boundary condition and the reynolds number)
- u_velocity and v_velocity are the x and y components of the velocity field respectively (2. and 3. images)
'''
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image

def open_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    return img

def gen_mask(Re, dimension):
    mask = np.zeros((dimension, dimension), dtype=np.float32)
    encoded_Re = (Re - 100) / 3900
    mask[:, :] = encoded_Re
    mask[-1, :] = mask[:, 0] = mask[:, -1] = 1
    return mask

def generate_input(Re, ts):
    u = open_image(f"../../data/{Re:.1f}/u_{ts:.2f}.png")
    v = open_image(f"../../data/{Re:.1f}/v_{ts:.2f}.png")
    dimension = u.shape[1]
    mask = gen_mask(Re, dimension)
    return np.stack([mask, u[0], v[0]], axis=0, dtype=np.float64)

def generate_output(Re):
    p = open_image(f"../../data/{Re:.1f}/p_ss.png")
    u = open_image(f"../../data/{Re:.1f}/u_ss.png")
    v = open_image(f"../../data/{Re:.1f}/v_ss.png")
    return np.stack([p[0], u[0], v[0]], axis=0, dtype=np.float64)


class ConvLSTM2DDataset(Dataset):
    def __init__(self, mode="train"):
        self.REs = np.arange(100., 4000.0, 100.)
        self.timesteps = np.arange(0., 9.95, 0.05)
        #self.timesteps = np.arange(0., 9.95, 0.25)
        self.one_sample_timestep = 1        # how many images are stacked together to form a single input sample (as a time series)
        self.predict_timestep = 1           # how many images are predicted
        
        self.totalLength = len(self.REs) * len(self.timesteps)
        self.inputs = np.empty((self.totalLength, self.one_sample_timestep, 3, 128, 128), dtype=np.float64)
        self.targets = np.empty((self.totalLength, self.predict_timestep, 3, 128, 128), dtype=np.float64)
        
        for i in range(self.totalLength):
            RE = random.choice(self.REs)
            ts_index = random.randint(0, len(self.timesteps) - self.one_sample_timestep)
            for j in range(self.one_sample_timestep):
                self.inputs[i, j] = generate_input(RE, self.timesteps[ts_index+j])
            for j in range(self.predict_timestep):
                self.targets[i, j] = generate_output(RE)
                
        if mode == "train":
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]
    
    def __len__(self):
        return self.totalLength
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ValiDataset(ConvLSTM2DDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]