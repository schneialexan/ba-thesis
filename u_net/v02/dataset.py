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

def gen_mask(Re):
    # generate mask with the boundary conditions and RE encoded into the image
    mask = np.zeros((128, 128, 3), dtype=np.uint8)
    # mask is the Re number where min is 0 and max is 4000 scaled to 0-255
    encoded_RE_color = (int(Re / 4000 * 255), 255, 0)
    mask[:, :, :] = encoded_RE_color
    # walls on the left, right and bottom
    border_color = (255, 255, 255)
    mask[-1, :, :] = border_color # bottom
    mask[:, 0, :] = border_color # left
    mask[:, -1, :] = border_color # right
    return mask

def generate_input(Re, ts):
    mask = gen_mask(Re).transpose(2, 0, 1)
    u = open_image(f"../../data/{Re:.1f}/u_{ts:.2f}.png")
    v = open_image(f"../../data/{Re:.1f}/v_{ts:.2f}.png")
    return np.stack([mask[0], u[0], v[0]], axis=0)

def generate_output(Re):
    p = open_image(f"../../data/{Re:.1f}/p_ss.png")
    u = open_image(f"../../data/{Re:.1f}/u_ss.png")
    v = open_image(f"../../data/{Re:.1f}/v_ss.png")
    return np.stack([p[0], u[0], v[0]], axis=0)

class TrainDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST
        
        self.REs = np.arange(100., 4000.0, 50.)
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

    #  reverts normalization 
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] = a[0,:,:] / (1.0/self.max_targets_0)
        a[1,:,:] = a[1,:,:] / (1.0/self.max_targets_1)
        a[2,:,:] = a[2,:,:] / (1.0/self.max_targets_2)

        if makeDimLess:
            a[0,:,:] = a[0,:,:] * v_norm**2
            a[1,:,:] = a[1,:,:] * v_norm
            a[2,:,:] = a[2,:,:] * v_norm
        return a

# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TrainDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

