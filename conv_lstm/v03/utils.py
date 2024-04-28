import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def normalize_image(data):
    min_value = np.min(data)
    max_value = np.max(data)
    data -= min_value
    data /= max_value
    return data


def fullImageOut(filename, _inputs, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
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