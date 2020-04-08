from __future__ import print_function
import numpy as np
import pandas as pd
import os
import gc
import cv2
import argparse
import random
import shutil
from scipy.stats import skewnorm
import math

def place_object(image, obj, loc_x=-1, loc_y=-1, allow_crop=True, noise=True, noise_amp=0.1):
    """
    Places obj into image, in a random location. Optionally adds Poisson
    (shot) noise.
    
    Parameters
    ----------
        image : np.array
            input (base) image
        obj : np.array
            object to place
        allow_crop : bool
            whether to allow partial placements (e.g. near edges)
        noise : bool
            add Poisson noise
        noise_amp : float
            noise amplitude

    Returns
    -------
        np.array
            image with object added
        tuple
            coordinates of object

    """
    
    height, width = image.shape
    o_height, o_width = obj.shape
    
    if loc_x == -1 and loc_y == -1:
        if allow_crop:
            loc_x = random.randint(0, width-1)
            loc_y = random.randint(0, height-1)
        else:
            loc_x = random.randint(o_width/2, width-1-o_width/2)
            loc_y = random.randint(o_height/2, height-1-o_height/2)
        
    pad_x = math.ceil(o_width/2)
    pad_y = math.ceil(o_height/2)
    
    loc_x += pad_x
    loc_y += pad_y

    image_pad = np.zeros((height+2*pad_y, width+2*pad_x))
    image_pad[pad_y:-pad_y, pad_x:-pad_x] = image

    lim_left = int(loc_x-o_width/2)
    lim_right = int(loc_x+o_width/2)

    lim_top = int(loc_y-o_height/2)
    lim_bottom = int(loc_y+o_height/2)
    
    if noise:
        obj += noise_amp*np.random.poisson(0.1, size=obj.shape)
    
    image_pad[lim_top:lim_bottom,lim_left:lim_right] += obj
            
    return image_pad[pad_y:-pad_y, pad_x:-pad_x], (loc_x-pad_x, loc_y-pad_y)

def skew_gaussian(size=9, amp_mu=5, amp_std=2, skew_x=3, skew_y=3):
    """
    Generate a random skewed 2D Gaussian. This code uses the dot product of
    two skew normal distributions which is probably not statistically
    meaningful, but it's a cheap way of generating skewed distributions.
    
    
    Parameters
    ----------
        size : int
            output size, should be odd
        amp_mu : float
            output amplitude mean
        amp_std : np.array
            output amplitude stdev
        skew_x:
            max skew to generate in x
        skew_y:
            max skew to generate in y
    Returns
    -------
        np.array
            generated distribution

    """
    
    if not size % 2:
        raise ValueError("Size must be odd!")
    
    # Generate 2 1D distributions
    a = -skew_x/2+skew_x*random.random()
    b = -skew_y/2+skew_y*random.random()

    # Size of blob
    x = np.linspace(-0.8*max(skew_x, skew_y),0.8*max(skew_x, skew_y), size)
    
    # Generate a 2D skewed distribution
    blob = np.dot(skewnorm.pdf(x, a).reshape((-1,1)),  skewnorm.pdf(x, b).reshape((1,-1)))
            
    # Normalise
    blob /= np.max(blob)
    
    # Scale by random amplitude
    amp = np.random.normal(amp_mu, amp_std)
    blob *= amp*(1.0/np.max(blob))
    
    return blob

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_blobs", type=int, required=True, help="max blobs to place")
    ap.add_argument("-s", "--size", type=float, required=True, help="max blob size")
    ap.add_argument("-a", "--amplitude", type=float, required=True, help="blob brightness")
    ap.add_argument("-x", "--skew_x", default=1, type=float, required=True, help="skew in x")
    ap.add_argument("-y", "--skew_y", default=3, type=float, required=True, help="skew in y")
    ap.add_argument("-p", "--prob", type=float, required=True, help="probability of adding blob", default=0.5)
    ap.add_argument("-i", "--class_id", type=int, required=True, help="object class ID")

    args = vars(ap.parse_args())

    imgs = []
    cwd = os.getcwd()

    # Load image names
    for file in os.listdir(cwd):
        if (file.endswith(".png") or file.endswith(".jpg")):
            imgs.append(file)
            
    # Output data folder
    new_folder = os.path.join(cwd, 'blob_aug')
    os.makedirs(new_folder, exist_ok=True)

    for file in imgs:
            
        # Important so that we can preserve radiometric if necessary
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        # Generate blobs
        num_blobs = random.randint(0, args.num_blobs)
        bboxes = []
        
        for i in range(num_blobs):
            if random.random() > args.p:

                # Place an blob
                obj = skew_gaussian(size=args.size, skew_x=1, skew_y=3, amp_mu=args.amplitude)
                img, coord = place_object(img, obj)

                x, y = coord
                height, width = img.shape
                centre_x -= args.size/4
                centre_y -= args.object_size/4
                obj_width = args.object_size
                obj_height = args.object_size

                bboxes.append("{} {} {} {} {}\n".format(args.class_id, centre_x/width, centre_y/height, obj_width/width, obj_height/height))

        # Write new image
        cv2.imwrite(os.path.join(new_folder, file), img)

        # Copy old label file, if exists
        label_file = file[:-4] + ".txt"
        if os.path.exists(os.path.join(cwd, label_file)):
            shutil.copy2(os.path.join(cwd, label_file), os.path.join(new_folder, label_file))

        # Write boxes (append and create file if necessary)
        for box in bboxes:
            with open(os.path.join(new_folder, label_file), "a+") as f:
                f.write(box)
