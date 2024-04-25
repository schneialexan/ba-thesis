
import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def computeLR(i,epochs, minLR, maxLR):
    if i < epochs*0.5:
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR-minLR)*f


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    
    s = outputs.shape[1] # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ( (s+10)*3, s*2) , color=(255,255,255) )
        BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )

    for i in range(3):
        #outputs[i] = np.flipud(outputs[i].transpose())
        #targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] = targets[i] - min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] = targets[i] / max_value
        else: # from -1,1 to 0,1
            outputs[i] -= -1.
            targets[i] = targets[i] - -1.
            outputs[i] /= 2.
            targets[i] = targets[i] / 2.

        if not saveMontage:
            suffix = ""
            if i==0:
                suffix = "_pressure"
            elif i==1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512,512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            # Color image (target and prediction)
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*1))

            # Black and white (1. target, 2. prediction, 3. error)
            im = Image.fromarray(targets[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*0))           
            im = Image.fromarray(outputs[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*1))
            imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            BW_im.paste(imE, ( (s+10)*i, s*2))

    if saveMontage:
        new_im.save(filename + ".png")
        BW_im.save( filename + "_bw.png")
        
def imOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    filename = filename + "_plt"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    
    ax1.imshow(targets[0], cmap='viridis')
    ax2.imshow(outputs[0], cmap='viridis')
    ax1.set_title('Target')
    ax2.set_title('Prediction')
    plt.savefig(filename + ".png")
    
    
def imsOut(filename, _inputs, _outputs, _targets):
    filename = filename + "p_u_v"
    inputs = np.copy(_inputs)
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        # first the inputs
        axes[0, i].imshow(inputs[i].transpose(1, 2, 0), cmap='viridis')
        axes[0, i].set_title('Input')
        # then the outputs (predictions) 
        axes[1, i].imshow(outputs[i].transpose(1, 2, 0), cmap='viridis')
        axes[1, i].set_title('Prediction')
        # and finally the targets
        axes[2, i].imshow(targets[i].transpose(1, 2, 0), cmap='viridis')
        axes[2, i].set_title('Target')        
    plt.savefig(filename + ".png")
    
def writeLog(filename, epoch, i, lossL1viz):
    logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch+1, i, lossL1viz)
    print(logline)
    with open(filename, 'a') as f:
        f.write(logline)



def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
    
    