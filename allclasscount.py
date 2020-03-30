import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--filetype", type=str, required=True, help="input filetype")
#args = vars(ap.parse_args())

#filetype = args["filetype"]

## Open a list of file names in the current work directory cwd.

imgs = []
cwd = os.getcwd()

## Add .png files to the list of file names.

for file in os.listdir(cwd):
	if (file.endswith(".png") or file.endswith(".jpg")):
		imgs.append(cwd + "/" + file)

## Read in the label files and compute the number of each class.

classes = []

for img in imgs:

	try:

		labels = pd.read_csv(img[:-4] + ".txt", sep = " ", header = None).values

		lab_row = labels.shape[0]
	
		for i in range(0, lab_row):
			if not (labels[i,1] == 0 and labels[i,2] == 0 and labels[i,3] == 0 and labels[i,4] == 0):
				classes.append(labels[i,0])
	except:
		continue

max_class = int(np.max(classes))

labs = []

for i in range(0, max_class+1):
	class_num = sum(s == i for s in classes)
	labs.append(class_num)

print(labs)


#filename = args['file']

#with open(filename, 'w') as f:
#	for i in imgs:
#		f.write("%s\n" % i)
