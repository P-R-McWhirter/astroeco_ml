import cv2
import numpy as np
import os
import csv
import pandas as pd

imgs = []
cwd = os.getcwd()

for file in os.listdir(cwd):
	if file.endswith(".png"):
		imgs.append(cwd + "/" + file)

imgs = np.sort(imgs)

for img in imgs:
	labels = pd.read_csv(img[:-4] + ".txt", sep = " ", header = None)
	labels.columns = ['class','x','y','w','h']
	cond = labels['class'] < 2
	labels = labels[cond].values

	os.remove(img[:-4] + ".txt")
	np.savetxt(img[:-4] + ".txt", labels, delimiter = " ", fmt = "%i %1.6f %1.6f %1.6f %1.6f")
	
	
	
