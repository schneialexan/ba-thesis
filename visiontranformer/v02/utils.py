
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


def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def fullImageOut(filename, _inputs, _targets, _outputs, saveTargets=False, normalize=False, saveMontage=True):
    inputs = np.copy(_inputs)
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    
    inputs = normalize_image(inputs)
    outputs = normalize_image(outputs)
    targets = normalize_image(targets)
    
    s = outputs.shape[1] # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )
        BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), layout='constrained')
    
    # iterate over the 3 channels (pressure, velX, velY)
    for i in range(3): 
        min_value = min(np.min(outputs[i]), np.min(targets[i]), np.min(inputs[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]), np.max(inputs[i]))
        
        if normalize:
            inputs[i] -= min_value
            outputs[i] -= min_value
            targets[i] = targets[i] - min_value
            max_value -= min_value
            inputs[i] /= max_value
            outputs[i] /= max_value
            targets[i] = targets[i] / max_value
        else: # from -1,1 to 0,1
            inputs[i] -= -1.
            outputs[i] -= -1.
            targets[i] = targets[i] - -1.
            inputs[i] /= 2.
            outputs[i] /= 2.
            targets[i] = targets[i] / 2.
        
        if saveMontage:
            prefix = ""
            if i==0:
                prefix = "Pressure"
            elif i==1:
                prefix = "Velocity X"
            else:
                prefix = "Velocity Y"            
            axes[0, i].imshow(inputs[i], cmap='viridis')
            axes[1, i].imshow(targets[i], cmap='viridis')
            axes[2, i].imshow(outputs[i], cmap='viridis')
            
            imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            axes[3, i].imshow(imE)
            
            axes[0, i].set_title('Input: ' + prefix if i!=0 else 'Mask (BC + RE)')
            axes[1, i].set_title('Target: ' + prefix)
            axes[2, i].set_title('Output: ' + prefix)
            axes[3, i].set_title('Error (Target - Output)')
            
    plt.colorbar(cm.ScalarMappable(cmap='gray'), ax=axes[3:])
    if saveMontage:
        plt.savefig(filename + "_change.png")
    
def writeLog(filename, epoch, i, lossL1viz):
    logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch+1, i, lossL1viz)
    with open(filename, 'a') as f:
        f.write(logline)


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
    
    