import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, required=True, help="output filename")
args = vars(ap.parse_args())

## Open a list of file names in the current work directory cwd.

imgs = []
cwd = os.getcwd()

## Add .png files to the list of file names.

for file in os.listdir(cwd):
	if (file.endswith(".png") or file.endswith(".jpg")):
		imgs.append(cwd + "/" + file)

## For each of the images, save the file paths to argument name.

filename = args['file']

with open(filename, 'w') as f:
	for i in imgs:
		f.write("%s\n" % i)
