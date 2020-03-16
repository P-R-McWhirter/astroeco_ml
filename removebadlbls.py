# Imports
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import gc
import itertools
import cv2

gc.enable()

files = []
cwd = os.getcwd()

for file in os.listdir(cwd):
    if file.endswith(".txt"):
        files.append(os.path.join(cwd, file))
        
files = np.sort(files)

for file in files:
    try:
        labels = pd.read_csv(file, sep = " ", header = None)
        labels.columns = ['class','x','y','w','h']
        cond = (labels['w'] != 0) & (labels['h'] != 0)
        labels = labels[cond].values
	
        os.remove(file)
        np.savetxt(file, labels, delimiter = " ", fmt = "%i %1.6f %1.6f %1.6f %1.6f")
    except:
        continue
