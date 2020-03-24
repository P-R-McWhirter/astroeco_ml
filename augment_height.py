# Imports
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import gc
import cv2
import argparse

gc.enable()

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--start", type=float, required=True, help="starting height of data")
ap.add_argument("-e", "--end", type=float, required=True, help="ending height of data")
args = vars(ap.parse_args())

start = args["start"]
end = args["end"]

ratio = start/end

color = [0, 0, 0]

imgs = []
cwd = os.getcwd()

altaug_folder = os.path.join(cwd, 'alt_aug')

if not os.path.exists(altaug_folder):

    os.makedirs(altaug_folder)

for file in os.listdir(cwd):
    if (file.endswith(".png") or file.endswith(".jpg")):
        imgs.append(file)
        
new_folder = os.path.join(altaug_folder, 'altaug_data_' + str(int(start)) + '_' + str(int(end)))
        
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

for file in imgs:
        
    img = cv2.imread(file)

    newsize_x = int(img.shape[0] * ratio)
    newsize_y = int(img.shape[1] * ratio)

    resize = cv2.resize(img, dsize=(newsize_y, newsize_x), interpolation=cv2.INTER_CUBIC)

    delta_w = img.shape[1] - resize.shape[1]
    delta_h = img.shape[0] - resize.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    cv2.imwrite(new_folder + '/' + file, new_im)
    
    try:
        labels = pd.read_csv(cwd + "/" + file[:-4] + ".txt", sep = " ", header = None).values
	
        labels[:,1] = (labels[:,1] - 0.5) * ratio + 0.5
        labels[:,2] = (labels[:,2] - 0.5) * ratio + 0.5
        labels[:,3] = labels[:,3] * ratio
        labels[:,4] = labels[:,4] * ratio

        np.savetxt(new_folder + '/' + file[:-4] + ".txt", labels, delimiter = ' ', fmt='%i %1.6f %1.6f %1.6f %1.6f')
    
    except:
        np.savetxt(new_folder + '/' + file[:-4] + ".txt", np.array([]), delimiter = ' ')
    
