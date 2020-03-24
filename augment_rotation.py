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
ap.add_argument("-a", "--angle", type=int, required=True, help="integer angle to rotate by in degrees")
args = vars(ap.parse_args())

rot_angle = int(args["angle"])

imgs = []
cwd = os.getcwd()

rotaug_folder = os.path.join(cwd, 'rot_aug')

if not os.path.exists(rotaug_folder):

    os.makedirs(rotaug_folder)

for file in os.listdir(cwd):
    if (file.endswith(".png") or file.endswith(".jpg")):
        imgs.append(file)



def get_corners(bboxes):
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners



def rotate_box(corners, M):

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated




def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final



for file in imgs:
        
    img = cv2.imread(file)
    
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    try:

        labels = pd.read_csv(cwd + "/" + file[:-4] + ".txt", sep = " ", header = None).values
    
        bbox_x1 = (labels[:,1] - (labels[:,3] / 2.0)) * w
        bbox_y1 = (labels[:,2] - (labels[:,4] / 2.0)) * h
        bbox_x2 = (labels[:,1] + (labels[:,3] / 2.0)) * w
        bbox_y2 = (labels[:,2] + (labels[:,4] / 2.0)) * h
        
        bbox_all = np.column_stack((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
    
        corners = get_corners(bbox_all)
        
        label_process = True
        
    except:
        
        label_process = False
        
        
        
    for angle in np.arange(0, 360, rot_angle):
        
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        
        rotated = cv2.warpAffine(img, M, (w,h))
        
        cv2.imwrite(os.path.join(rotaug_folder, file[:-4] + '_' + str(angle) + file[-4:]), rotated)
        
        if label_process == True:
    
            rot_coords = rotate_box(corners, M)
        
            rot_boxes = get_enclosing_box(rot_coords)
        
            rot_labels_w = (rot_boxes[:,2] - rot_boxes[:,0])
            rot_labels_h = (rot_boxes[:,3] - rot_boxes[:,1])
            rot_labels_x = rot_boxes[:,0] + (rot_labels_w // 2)
            rot_labels_y = rot_boxes[:,1] + (rot_labels_h // 2)
        
            rot_labels_x = rot_labels_x / w
            rot_labels_y = rot_labels_y / h
            rot_labels_w = rot_labels_w / w
            rot_labels_h = rot_labels_h / h
        
            rot_labels = np.column_stack((labels[:,0], rot_labels_x, rot_labels_y, rot_labels_w, rot_labels_h))
    
            np.savetxt(os.path.join(rotaug_folder, file[:-4] + '_' + str(angle) + ".txt"), rot_labels, delimiter = ' ', fmt='%i %1.6f %1.6f %1.6f %1.6f')
        
        else:
            
            np.savetxt(os.path.join(rotaug_folder, file[:-4] + '_' + str(angle) + ".txt"), np.array([]), delimiter = ' ')
