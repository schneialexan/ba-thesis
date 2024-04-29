from sre_constants import RANGE
from torch.utils.data import Dataset
import numpy as np
from os import listdir
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
    return np.stack([mask, u[0], v[0]], axis=0)

def generate_output(Re):
    p = open_image(f"../../data/{Re:.1f}/p_ss.png")
    u = open_image(f"../../data/{Re:.1f}/u_ss.png")
    v = open_image(f"../../data/{Re:.1f}/v_ss.png")
    return np.stack([p[0], u[0], v[0]], axis=0)

class TrainDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, mode=TRAIN):
        self.mode = mode
        
        self.REs = np.arange(100., 4000.0, 100.)
        self.timesteps = np.arange(0., 9.95, 0.05)       # timestep 10. is steady state

        # load & normalize data
        self.totalLength = len(self.REs) * len(self.timesteps)
        self.inputs  = np.empty((self.totalLength, 3, 128, 128))
        self.targets = np.empty((self.totalLength, 3, 128, 128))
        
        for i in range(self.totalLength):
            '''
            input:  starting picture (y_vel = 0, x_vel on the lid als 1 und rest als 0, mask, ränder als 1 und rest als 0)  -> Rey 100-4000, bei x_vel einbinden oder bei der maske einbinden (mit den ränder) 
            output: steady state, a.k.a. the target picture
            '''
            Re = random.choice(self.REs)
            ts = random.choice(self.timesteps)
            self.inputs[i] = generate_input(Re, ts)             # zufalls reynold und zufalls zeitschritt und [mask, u, v]
            self.targets[i] = generate_output(Re)               # steady state, von dieser RE, letzter Zeitschritt, (evtl.pressure), u, v
            

        if not self.mode==self.TEST:
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

class ValiDataset(TrainDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

