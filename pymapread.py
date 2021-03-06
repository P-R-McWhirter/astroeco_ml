# Imports
from __future__ import print_function
import numpy as np
#import pandas as pd
import os
import gc
import itertools
import argparse
import subprocess
import re
import csv

gc.enable()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prefix", type=str, required=True, help="prefix for selecting darknet models")
ap.add_argument("-d", "--datapath", type=str, required=True, help="object relative data path for darknet")
ap.add_argument("-c", "--cfgpath", type=str, required=True, help="config relative data path for darknet")
ap.add_argument("-o", "--output", type=str, required=True, help="output csv file for results")
ap.add_argument("-n", "--darknet", type=str, required=True, help="path to the darknet executable folder")
ap.add_argument("-i", "--iou_thresh", type=float, required=True, help="IoU threshold value")
args = vars(ap.parse_args())

prefix = args["prefix"]
datapath = args["datapath"]
cfgpath = args["cfgpath"]
output = args["output"]
darknet = args["darknet"]
iou_thresh = args["iou_thresh"]

filetype = '.weights'

models = []
cwd = os.getcwd()

cwd_split = os.path.split(cwd)

cwd_name = cwd_split[len(cwd_split)-1]

for file in os.listdir(cwd):
    if file.startswith(prefix):
        if file.endswith(filetype):
            models.append(file)
        
models = sorted(models)

mods_len = len(models)

os.chdir(darknet)
	
first_result = subprocess.check_output(['./darknet detector map ' + datapath + ' ' + cfgpath + ' ' + os.path.join(cwd, models[0]) + ' -iou_thresh ' + str(iou_thresh)], shell=True)

first_result = re.split('\t|\n|\r', first_result.decode())

first_result = [models[0]] + first_result

matchers = ['rank']

first_result = [s for s in first_result if not any(xs in s for xs in matchers)]

with open(output, 'w+', newline='') as csvfile:
    modwriter = csv.writer(csvfile, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    modwriter.writerow(first_result)
    csvfile.close()

for mod_num in range(1, len(models)):

    new_result = subprocess.check_output(['./darknet detector map ' + datapath + ' ' + cfgpath + ' ' + os.path.join(cwd, models[mod_num]) + ' -iou_thresh ' + str(iou_thresh)], shell=True)

    new_result = re.split('\t|\n|\r', new_result.decode())
	
    new_result = [models[mod_num]] + new_result

    new_result = [s for s in new_result if not any(xs in s for xs in matchers)]

    with open(output, 'a+', newline='') as csvfile:
        modwriter = csv.writer(csvfile, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        modwriter.writerow(new_result)
        csvfile.close()
