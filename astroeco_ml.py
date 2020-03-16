###################################################################

####    Astroecology Machine Learning pipeline using YOLOv3    ####

####              Ross McWhirter --- March 2020                ####

###################################################################

import numpy as np
import argparse
import os
import csv
from subprocess import check_output, Popen, PIPE, DEVNULL
from shutil import rmtree

# Collect required arguments to determine which processes are needed.

ap = argparse.ArgumentParser()
ap.add_argument("--darknet", type=str, required=False, help="Over-ride path to darknet folder.")
ap.add_argument("--ultra", type=str, required=False, help="Over-ride path to ultralytics folder.")
ap.add_argument("--mode", type=str, required=True, help="Input 'train', 'eval' or 'detect' to select mode.")

# Train arguments.

ap.add_argument("--train_folder", type=str, required=False, help="Folder containing images and labels for the training dataset.")
ap.add_argument("--val_folder", type=str, required=False, help="Folder containing images and labels for the validation dataset.")
ap.add_argument("--model_type", type=str, required=False, help="Model type for training. Can be either 'yolov3', 'yolov3-spp', 'yolov3-tiny' or 'yolov3-tiny-3l'")
ap.add_argument("--gpu_id", type=int, required=False, default=0, help="GPU ID for training from 0+. Defaults to 0 if blank.")
ap.add_argument("--subdivisions", type=int, required=False, default=16, help="Number of subdivisions from batch of 64. Defaults to 16, increase/decrease depending on GPU RAM availability.")
ap.add_argument("--model_dim", type=int, required=False, help="Model input image width and height in pixels. Must be a multiple of 32.")
ap.add_argument("--num_of_epochs", type=float, required=False, help="Number of epochs to train the model.")
ap.add_argument("--data_folder", type=str, required=False, help="Folder to output the data files of the trained model. Note: this will overwrite data files in this folder.")
ap.add_argument("--model_folder", type=str, required=False, help="Folder to output the weights files of the trained model. Note: this may overwrite weights files in this folder.")

# Detection arguments.

ap.add_argument("--cfg", type=str, required=False, help="Config file of existing detection model.")
ap.add_argument("--weights", type=str, required=False, help="Weights (model) file of existing detection model.")
ap.add_argument("--obj_names", type=str, required=False, help="Object names file (lists the classes).")
ap.add_argument("--input", type=str, required=False, help="Input folder of images to be classified with existing detection model.")
ap.add_argument("--conf_thres", type=float, required=False, help="Confidence threshold probability between 0 and 1.")
ap.add_argument("--make_video", type=bool, required=False, default=False, help="Boolean flag (True/False) to generate a video from final classifications.")
ap.add_argument("--video_framerate", type=float, required=False, help="Framerate of output video from final classifications.")
ap.add_argument("--output", type=str, required=False, help="Output folder of images classified with existing detection model.")

# Parse the arguments.

args = vars(ap.parse_args())

# Get the current work directory

cwd = os.getcwd()

# Tell the world about yourself

print("****************************************************")

print("Astroecology Machine Learning pipeline using YOLOv3.")

print("      Written by Ross McWhirter in March 2020.")

print("****************************************************")

# Check to make sure a valid runtime mode is entered otherwise exit.

mode = args['mode']

if (mode != 'train' and mode != 'eval' and mode != 'detect'):
    raise NameError("'" + mode + "' is not a valid mode. Exiting...")

# Get the darknet path. If none is entered, use the default on the astroecology machine.

darknet_path = args['darknet']

if darknet_path is None:
    darknet_path = '/home/astroecology/Documents/darknet'

# Get the ultralytics path. If none is entered, use the default on the astroecology machine.

ultra_path = args['ultra']

if ultra_path is None:
    ultra_path = '/home/astroecology/Documents/yolov3'

# Check to see if darknet executable exists in the darknet folder otherwise exit.

print("Checking presence of darknet executable...")

darknet_present = os.path.isfile(os.path.join(darknet_path, 'darknet'))

if darknet_present == False:
    raise NameError("Darknet executable not present in darknet target folder. Exiting...")
else:
    print("Done.")

# Then check to see if ultralytics python script exists in the ultralytics folder otherwise exit.

print("Checking presence of ultralytics detect.py script...")

ultra_present = os.path.isfile(os.path.join(ultra_path, 'detect.py'))

if ultra_present == False:
    raise NameError("Ultralytics detect.py script not present in ultralyics target folder. Exiting...")
else:
    print("Done.")

# Define functions for the three operating modes of this script.

def astroeco_ml_train(args, cwd, darknet_path):

    print("********************************************")

    print("This is the function for training ML models.")

    print("********************************************")

    os.chdir(cwd)

    # Check to make sure a valid model type is entered otherwise exit.

    print("Identifying model type for darknet training process...")

    model_type = args['model_type']

    if model_type is None:
        raise NameError("No model type supplied, use the --model_type argument to select a model type for training. Exiting...")
    else:
        if (model_type != 'yolov3' and model_type != 'yolov3-spp' and model_type != 'yolov3-tiny' and model_type != 'yolov3-tiny-3l'):
            raise NameError("'" + model_type + "' is not a valid model type. Exiting...")
        else:
            print("Model type: '" + model_type + "' selected for training.")

    # Open object names filepath.

    print("Opening object names file for darknet model...")

    obj_names_path = args['obj_names']

    if obj_names_path is None:
        raise NameError("No object names file entered, use the --obj_names argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Load in the training and validation data.

    print("Opening folder containing training data...")

    train_folder = args['train_folder']

    if train_folder is None:
        raise NameError("No training set folder path, use the --train_folder argument to supply the path. Exiting...")
    else:
        if not os.path.exists(os.path.join(cwd, train_folder)):
            raise NameError("Supplied training set folder path does not exist. Exiting...")
        else:
            print("Done.")

    print("Opening folder containing validation data...")

    val_folder = args['val_folder']

    if val_folder is None:
        raise NameError("No validation set folder path, use the --val_folder argument to supply the path. Exiting...")
    else:
        if not os.path.exists(os.path.join(cwd, val_folder)):
            raise NameError("Supplied validation set folder path does not exist. Exiting...")
        else:
            print("Done.")

    #######################################################################

    ## Consider adding more data augmentation (altitude, rotation) here. ##

    #######################################################################

    # Load in the output model 'backup' folder.

    print("Opening output folder for trained weights files...")

    model_folder = args['model_folder']

    if model_folder is None:
        raise NameError("No output model folder path, use the --model_folder argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Create the output model folder if it doesn't exist

    if not os.path.exists(os.path.join(cwd, model_folder)):

        os.makedirs(os.path.join(cwd, model_folder))

    # Setup the folder for the obj.data file.

    print("Opening output folder to store training obj.data files...")

    data_folder = args['data_folder']

    if data_folder is None:
        raise NameError("No output data folder path, use the --data_folder argument to supply the path. Exiting...")
    else:
        print("Done.")

    if not os.path.exists(os.path.join(cwd, data_folder)):

        os.makedirs(os.path.join(cwd, data_folder))

    # Build the obj.data file based on the inputs we have so far.

    data_folder_path = os.path.join(cwd, data_folder)

    # First we must produce the train.txt and val.txt files. Fortunately we have a python script (makedatafiles.py) in this directory to do this.

    # We can also fix any labelling bugs from the Deeplabel export using removebadlbls.py

    print("Processing training data prior to model training...")

    os.chdir(os.path.join(cwd, train_folder))

    check_output(['python ' + os.path.join(cwd, 'removebadlbls.py')], shell = True)

    check_output(['python ' + os.path.join(cwd, 'makedatafiles.py') + ' -f train.txt'], shell = True)

    check_output(['mv train.txt ' + data_folder_path], shell = True)

    print("Done.")

    print("Processing validation data prior to model training...")

    os.chdir(os.path.join(cwd, val_folder))

    check_output(['python ' + os.path.join(cwd, 'removebadlbls.py')], shell = True)

    check_output(['python ' + os.path.join(cwd, 'makedatafiles.py') + ' -f val.txt'], shell = True)

    check_output(['mv val.txt ' + data_folder_path], shell = True)

    print("Done.")

    # Open the names file and determine the number of classes.

    obj_names = open(os.path.join(cwd, obj_names_path), "r")

    classes = len(obj_names.readlines())

    # Also fetch the training set size.

    training_set = open(os.path.join(data_folder_path, 'train.txt'), "r")

    training_set_size = len(training_set.readlines())

    if training_set_size == 0:
        raise NameError("Something has gone wrong reading the training set size. This is not automatically repairable. Exiting...")

    # Now we can build the obj.data file.

    os.chdir(data_folder_path)

    obj_data = os.path.join(data_folder_path, 'obj.data')

    datafile = open(obj_data, "w")

    datafile.write("classes = " + str(classes) + "\n")

    datafile.write("train = " + os.path.join(data_folder_path, 'train.txt') + "\n")

    datafile.write("valid = " + os.path.join(data_folder_path, 'val.txt') + "\n")

    datafile.write("names = " + os.path.join(cwd, obj_names_path) + "\n")
    
    datafile.write("backup = " + os.path.join(cwd, model_folder))

    datafile.close()

    # Get the number of epochs for training

    num_of_epochs = args['num_of_epochs']

    if num_of_epochs is None:
        print("No number of epochs supplied, use the --num_of_epochs argument to supply the path. Defaulting to 100...")
        num_of_epochs = 100.
    else:
        print("Model will be trained for " + str(num_of_epochs) + " epochs.")

    # Get the subdivision size.

    subdivisions = args['subdivisions']

    print("Using batch size of 64 split into " + str(subdivisions) + " subdivisions...")

    # Fetch the GPU-id for training.
   
    gpu_id = args['gpu_id']

    print("Using GPU " + str(gpu_id) + " for training...")

    # Get the model input size from the arguments.

    print("Getting model input dimensions...")

    model_dim = args['model_dim']

    if model_dim is None:
        print("No model input image width and height supplied, use the --model_dim argument to supply the path. Defaulting to 416...")
        model_dim = 416
    else:
        if np.abs(model_dim/32. - np.floor(model_dim/32.)) == 0.0:
            print("Model has an input width and height of " + str(model_dim) + "x" + str(model_dim) + " pixels.")
        else:
            raise NameError(str(model_dim) + " is not a multiple of 32, please re-enter an appropriate value. Exiting...")

    print("Done.")

    # Compute the number of filters in the penultimate layer.

    filters = 3 * (classes + 5)

    # Select the appropriate function for further model configuration.

    if model_type == 'yolov3':
        
        astroeco_ml_train_yolov3(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder)

    elif model_type == 'yolov3-spp':
        
        astroeco_ml_train_yolov3_spp(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder)

    elif model_type == 'yolov3-tiny':

        astroeco_ml_train_yolov3_tiny(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder)

    elif model_type == 'yolov3-tiny-3l':

        astroeco_ml_train_yolov3_tiny_3l(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder)

    else:

        raise NameError("'" + model_type + "' is not a valid model type. You managed to avoid the first catch, I am impressed. Exiting...")

def astroeco_ml_train_yolov3(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder):

    print("*************************************************")

    print("This is the function for training a YOLOv3 model.")

    print("*************************************************")

    # Before we do anything, make sure the pre-trained model for YOLOv3 is in the base darknet directory with the expected name.

    print("Checking to make sure pre-trained weights file for YOLOv3 'darknet53.conv.74' is present...")

    if not os.path.exists(os.path.join(cwd, darknet_path, 'darknet53.conv.74')):

        raise NameError("Pre-trained weights file for YOLOv3 'darknet53.conv.74' is missing from the darknet folder. Please copy this file to the correct location. Exiting...")

    else:

        print("Done.")

    # Quickly compute the max_batches and steps values from the epoch number.

    print("Computing the number of iterations required from the number of epochs (" + str(num_of_epochs) + ")...")

    max_batches = int(np.ceil(num_of_epochs * training_set_size / 64.))

    steps_80 = int(np.ceil(0.8 * max_batches))

    steps_90 = int(np.ceil(0.9 * max_batches))

    print("Done. The model will be trained for " + str(max_batches) + " iterations.")

    # Compute the anchors for the yolov3 model using the darknet k-means clustering command.

    print("Computing anchor points for YOLOv3 model...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    calc_anchors = Popen(['./darknet detector calc_anchors ' + obj_data + ' -num_of_clusters 9 -width ' + str(model_dim) + ' -height ' + str(model_dim)], stdin = PIPE, stdout = DEVNULL, shell = True)

    calc_anchors.communicate(input = b'\n')

    # Read the new anchors from the anchors.txt file in the darknet directory.

    anchor_file = open('anchors.txt', "r")

    anchors = anchor_file.readlines()[0]

    print("Done.")

    # We need to construct the config file for the yolov3 model and move back to cwd.

    print("Building yolov3 configuration file...")

    os.chdir(cwd)

    obj_cfg = os.path.join(data_folder_path, 'yolov3.cfg')

    datafile = open(obj_cfg, "w")

    datafile.write("[net]\n")
    datafile.write("batch=64\n")
    datafile.write("subdivisions=" + str(subdivisions) + "\n")
    datafile.write("width=" + str(model_dim) + "\n")
    datafile.write("height=" + str(model_dim) + "\n")
    datafile.write("channels=3\n")
    datafile.write("momentum=0.9\n")
    datafile.write("decay=0.0005\n")
    datafile.write("angle=0\n")
    datafile.write("saturation=1.5\n")
    datafile.write("exposure=1.5\n")
    datafile.write("hue=.1\n")
    datafile.write("\n")
    datafile.write("learning_rate=0.001\n")
    datafile.write("burn_in=1000\n")
    datafile.write("max_batches=" + str(max_batches) + "\n")
    datafile.write("policy=steps\n")
    datafile.write("steps=" + str(steps_80) + "," + str(steps_90) + "\n")
    datafile.write("scales=.1,.1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("######################\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 6,7,8\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 61\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 3,4,5\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 36\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 0,1,2\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")

    datafile.close()

    print("Done.")

    # Now train the model using darknet as all the files are in place.

    print("Training darknet model using generated configuration files. This will likely take some time...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    check_output(['./darknet detector train ' + obj_data + ' ' + obj_cfg + ' darknet53.conv.74 -gpus ' + str(gpu_id)], shell = True)

    print("Done.")

    print("Moving training graph to output model folder...")

    check_output(['mv chart.png ' + os.path.join(cwd, model_folder)], shell = True)

    os.chdir(cwd)

    print("Done.")

    print("Training process complete. If you wish to evaluate the trained model, use the --mode eval option.")

def astroeco_ml_train_yolov3_spp(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder):

    print("*****************************************************")

    print("This is the function for training a YOLOv3-SPP model.")

    print("*****************************************************")

    # Before we do anything, make sure the pre-trained model for YOLOv3 is in the base darknet directory with the expected name.

    print("Checking to make sure pre-trained weights file for YOLOv3 'darknet53.conv.74' is present...")

    if not os.path.exists(os.path.join(cwd, darknet_path, 'darknet53.conv.74')):

        raise NameError("Pre-trained weights file for YOLOv3 'darknet53.conv.74' is missing from the darknet folder. Please copy this file to the correct location. Exiting...")

    else:

        print("Done.")

    # Quickly compute the max_batches and steps values from the epoch number.

    print("Computing the number of iterations required from the number of epochs (" + str(num_of_epochs) + ")...")

    max_batches = int(np.ceil(num_of_epochs * training_set_size / 64.))

    steps_80 = int(np.ceil(0.8 * max_batches))

    steps_90 = int(np.ceil(0.9 * max_batches))

    print("Done. The model will be trained for " + str(max_batches) + " iterations.")

    # Compute the anchors for the yolov3 model using the darknet k-means clustering command.

    print("Computing anchor points for YOLOv3-SPP model...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    calc_anchors = Popen(['./darknet detector calc_anchors ' + obj_data + ' -num_of_clusters 9 -width ' + str(model_dim) + ' -height ' + str(model_dim)], stdin = PIPE, stdout = DEVNULL, shell = True)

    calc_anchors.communicate(input = b'\n')

    # Read the new anchors from the anchors.txt file in the darknet directory.

    anchor_file = open('anchors.txt', "r")

    anchors = anchor_file.readlines()[0]

    print("Done.")

    # We need to construct the config file for the yolov3 model and move back to cwd.

    print("Building yolov3-SPP configuration file...")

    os.chdir(cwd)

    obj_cfg = os.path.join(data_folder_path, 'yolov3-spp.cfg')

    datafile = open(obj_cfg, "w")

    datafile.write("[net]\n")
    datafile.write("batch=64\n")
    datafile.write("subdivisions=" + str(subdivisions) + "\n")
    datafile.write("width=" + str(model_dim) + "\n")
    datafile.write("height=" + str(model_dim) + "\n")
    datafile.write("channels=3\n")
    datafile.write("momentum=0.9\n")
    datafile.write("decay=0.0005\n")
    datafile.write("angle=0\n")
    datafile.write("saturation=1.5\n")
    datafile.write("exposure=1.5\n")
    datafile.write("hue=.1\n")
    datafile.write("\n")
    datafile.write("learning_rate=0.001\n")
    datafile.write("burn_in=1000\n")
    datafile.write("max_batches=" + str(max_batches) + "\n")
    datafile.write("policy=steps\n")
    datafile.write("steps=" + str(steps_80) + "," + str(steps_90) + "\n")
    datafile.write("scales=.1,.1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=2\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[shortcut]\n")
    datafile.write("from=-3\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("######################\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("### SPP ###\n")
    datafile.write("[maxpool]\n")
    datafile.write("stride=1\n")
    datafile.write("size=5\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers=-2\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("stride=1\n")
    datafile.write("size=9\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers=-4\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("stride=1\n")
    datafile.write("size=13\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers=-1,-3,-5,-6\n")
    datafile.write("\n")
    datafile.write("### End SPP ###\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=1024\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 6,7,8\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 61\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=512\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 3,4,5\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 36\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=256\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 0,1,2\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")

    datafile.close()

    print("Done.")

    # Now train the model using darknet as all the files are in place.

    print("Training darknet model using generated configuration files. This will likely take some time...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    check_output(['./darknet detector train ' + obj_data + ' ' + obj_cfg + ' darknet53.conv.74 -gpus ' + str(gpu_id)], shell = True)

    print("Done.")

    print("Moving training graph to output model folder...")

    check_output(['mv chart.png ' + os.path.join(cwd, model_folder)], shell = True)

    os.chdir(cwd)

    print("Done.")

    print("Training process complete. If you wish to evaluate the trained model, use the --mode eval option.")

def astroeco_ml_train_yolov3_tiny(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder):

    print("******************************************************")

    print("This is the function for training a Tiny-YOLOv3 model.")

    print("******************************************************")

    # Before we do anything, make sure the pre-trained model for Tiny-YOLOv3 is in the base darknet directory with the expected name.

    print("Checking to make sure pre-trained weights file for YOLOv3 'yolov3-tiny.conv.15' is present...")

    if not os.path.exists(os.path.join(cwd, darknet_path, 'yolov3-tiny.conv.15')):

        raise NameError("Pre-trained weights file for Tiny-YOLOv3 'yolov3-tiny.conv.15' is missing from the darknet folder. Please copy this file to the correct location. Exiting...")

    else:

        print("Done.")

    # Quickly compute the max_batches and steps values from the epoch number.

    print("Computing the number of iterations required from the number of epochs (" + str(num_of_epochs) + ")...")

    max_batches = int(np.ceil(num_of_epochs * training_set_size / 64.))

    steps_80 = int(np.ceil(0.8 * max_batches))

    steps_90 = int(np.ceil(0.9 * max_batches))

    print("Done. The model will be trained for " + str(max_batches) + " iterations.")

    # Compute the anchors for the yolov3 model using the darknet k-means clustering command.

    print("Computing anchor points for Tiny-YOLOv3 model...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    calc_anchors = Popen(['./darknet detector calc_anchors ' + obj_data + ' -num_of_clusters 6 -width ' + str(model_dim) + ' -height ' + str(model_dim)], stdin = PIPE, stdout = DEVNULL, shell = True)

    calc_anchors.communicate(input = b'\n')

    # Read the new anchors from the anchors.txt file in the darknet directory.

    anchor_file = open('anchors.txt', "r")

    anchors = anchor_file.readlines()[0]

    print("Done.")

    # We need to construct the config file for the yolov3 model and move back to cwd.

    print("Building tiny-yolov3 configuration file...")

    os.chdir(cwd)

    obj_cfg = os.path.join(data_folder_path, 'yolov3-tiny.cfg')

    datafile = open(obj_cfg, "w")

    datafile.write("[net]\n")
    datafile.write("batch=64\n")
    datafile.write("subdivisions=" + str(subdivisions) + "\n")
    datafile.write("width=" + str(model_dim) + "\n")
    datafile.write("height=" + str(model_dim) + "\n")
    datafile.write("channels=3\n")
    datafile.write("momentum=0.9\n")
    datafile.write("decay=0.0005\n")
    datafile.write("angle=0\n")
    datafile.write("saturation=1.5\n")
    datafile.write("exposure=1.5\n")
    datafile.write("hue=.1\n")
    datafile.write("\n")
    datafile.write("learning_rate=0.001\n")
    datafile.write("burn_in=1000\n")
    datafile.write("max_batches=" + str(max_batches) + "\n")
    datafile.write("policy=steps\n")
    datafile.write("steps=" + str(steps_80) + "," + str(steps_90) + "\n")
    datafile.write("scales=.1,.1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=16\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("###########\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 3,4,5\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=6\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 8\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 0,1,2\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=6\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")

    datafile.close()

    print("Done.")

    # Now train the model using darknet as all the files are in place.

    print("Training darknet model using generated configuration files. This will likely take some time...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    check_output(['./darknet detector train ' + obj_data + ' ' + obj_cfg + ' yolov3-tiny.conv.15 -gpus ' + str(gpu_id)], shell = True)

    print("Done.")

    print("Moving training graph to output model folder...")

    check_output(['mv chart.png ' + os.path.join(cwd, model_folder)], shell = True)

    os.chdir(cwd)

    print("Done.")

    print("Training process complete. If you wish to evaluate the trained model, use the --mode eval option.")

def astroeco_ml_train_yolov3_tiny_3l(cwd, darknet_path, data_folder_path, obj_data, num_of_epochs, gpu_id, subdivisions, model_dim, training_set_size, classes, filters, model_folder):

    print("**************************************************************")

    print("This is the function for training a 3-layer Tiny-YOLOv3 model.")

    print("**************************************************************")

    # Before we do anything, make sure the pre-trained model for Tiny-YOLOv3 is in the base darknet directory with the expected name.

    print("Checking to make sure pre-trained weights file for YOLOv3 'yolov3-tiny.conv.15' is present...")

    if not os.path.exists(os.path.join(cwd, darknet_path, 'yolov3-tiny.conv.15')):

        raise NameError("Pre-trained weights file for Tiny-YOLOv3 'yolov3-tiny.conv.15' is missing from the darknet folder. Please copy this file to the correct location. Exiting...")

    else:

        print("Done.")

    # Quickly compute the max_batches and steps values from the epoch number.

    print("Computing the number of iterations required from the number of epochs (" + str(num_of_epochs) + ")...")

    max_batches = int(np.ceil(num_of_epochs * training_set_size / 64.))

    steps_80 = int(np.ceil(0.8 * max_batches))

    steps_90 = int(np.ceil(0.9 * max_batches))

    print("Done. The model will be trained for " + str(max_batches) + " iterations.")

    # Compute the anchors for the yolov3 model using the darknet k-means clustering command.

    print("Computing anchor points for 3-layer Tiny-YOLOv3 model...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    calc_anchors = Popen(['./darknet detector calc_anchors ' + obj_data + ' -num_of_clusters 9 -width ' + str(model_dim) + ' -height ' + str(model_dim)], stdin = PIPE, stdout = DEVNULL, shell = True)

    calc_anchors.communicate(input = b'\n')

    # Read the new anchors from the anchors.txt file in the darknet directory.

    anchor_file = open('anchors.txt', "r")

    anchors = anchor_file.readlines()[0]

    print("Done.")

    # We need to construct the config file for the yolov3 model and move back to cwd.

    print("Building yolov3-tiny-3l configuration file...")

    os.chdir(cwd)

    obj_cfg = os.path.join(data_folder_path, 'yolov3-tiny-3l.cfg')

    datafile = open(obj_cfg, "w")

    datafile.write("[net]\n")
    datafile.write("batch=64\n")
    datafile.write("subdivisions=" + str(subdivisions) + "\n")
    datafile.write("width=" + str(model_dim) + "\n")
    datafile.write("height=" + str(model_dim) + "\n")
    datafile.write("channels=3\n")
    datafile.write("momentum=0.9\n")
    datafile.write("decay=0.0005\n")
    datafile.write("angle=0\n")
    datafile.write("saturation=1.5\n")
    datafile.write("exposure=1.5\n")
    datafile.write("hue=.1\n")
    datafile.write("\n")
    datafile.write("learning_rate=0.001\n")
    datafile.write("burn_in=1000\n")
    datafile.write("max_batches=" + str(max_batches) + "\n")
    datafile.write("policy=steps\n")
    datafile.write("steps=" + str(steps_80) + "," + str(steps_90) + "\n")
    datafile.write("scales=.1,.1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=16\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=32\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=64\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[maxpool]\n")
    datafile.write("size=2\n")
    datafile.write("stride=1\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=1024\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("###########\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=512\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 6,7,8\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -4\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 8\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=256\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 3,4,5\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -3\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[upsample]\n")
    datafile.write("stride=2\n")
    datafile.write("\n")
    datafile.write("[route]\n")
    datafile.write("layers = -1, 6\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("batch_normalize=1\n")
    datafile.write("filters=128\n")
    datafile.write("size=3\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("activation=leaky\n")
    datafile.write("\n")
    datafile.write("[convolutional]\n")
    datafile.write("size=1\n")
    datafile.write("stride=1\n")
    datafile.write("pad=1\n")
    datafile.write("filters=" + str(filters) + "\n")
    datafile.write("activation=linear\n")
    datafile.write("\n")
    datafile.write("[yolo]\n")
    datafile.write("mask = 0,1,2\n")
    datafile.write("anchors = " + anchors + "\n")
    datafile.write("classes=" + str(classes) + "\n")
    datafile.write("num=9\n")
    datafile.write("jitter=.3\n")
    datafile.write("ignore_thresh = .7\n")
    datafile.write("truth_thresh = 1\n")
    datafile.write("random=1\n")

    datafile.close()

    print("Done.")

    # Now train the model using darknet as all the files are in place.

    print("Training darknet model using generated configuration files. This will likely take some time...")

    # Move to the darknet folder.

    os.chdir(darknet_path)

    check_output(['./darknet detector train ' + obj_data + ' ' + obj_cfg + ' yolov3-tiny.conv.15 -gpus ' + str(gpu_id)], shell = True)

    print("Done.")

    print("Moving training graph to output model folder...")

    check_output(['mv chart.png ' + os.path.join(cwd, model_folder)], shell = True)

    os.chdir(cwd)

    print("Done.")

    print("Training process complete. If you wish to evaluate the trained model, use the --mode eval option.")

def astroeco_ml_eval():

    print("**********************************************")

    print("This is the function for evaluating ML models.")

    print("**********************************************")

    # Move work directory to the darknet folder.

    os.chdir(darknet_path)

def astroeco_ml_detect(args, cwd, ultra_path):

    print("*******************************************************************")

    print("This is the function for detecting objects from a trained ML model.")

    print("*******************************************************************")

    # Move work directory to the ultralytics folder.

    os.chdir(ultra_path)

    # Open config filepath.

    print("Opening configuration file for darknet model...")

    cfg_path = args['cfg']

    if cfg_path is None:
        raise NameError("No configuration file entered, use the --cfg argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Open weights filepath.

    print("Opening model weights file for darknet model...")

    weights_path = args['weights']

    if weights_path is None:
        raise NameError("No weights file entered, use the --weights argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Open object names filepath.

    print("Opening object names file for darknet model...")

    obj_names_path = args['obj_names']

    if obj_names_path is None:
        raise NameError("No object names file entered, use the --obj_names argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Open folder containing target images.

    print("Opening target image folder for classification...")

    input_path = args['input']

    if input_path is None:
        raise NameError("No input image folder path, use the --input argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Collecting confidence threshold.

    print("Reading confidence threshold from arguments...")

    conf_thres = args['conf_thres']

    if conf_thres is None:
        conf_thres = 0.25
        print("No confidence threshold supplied, use the --conf_thres argument to enter one. Defaulting to 0.25... Done.")
    else:
        print("Confidence threshold of " + str(conf_thres) + " supplied... Done.")

    # Use the object names file to construct a temporary data file for correctly labelling classes.

    # Open the names file and determine the number of classes.

    obj_names = open(obj_names_path, "r")

    classes = len(obj_names.readlines())

    # First create an empty.txt file for dummy inputs for the data file such as training and testing sets not required for inference.

    empty_file = os.path.join(cwd, 'empty.txt')

    check_output(['touch ' + empty_file], shell = True)

    # Next open a file writer to populate the required fields of the data file and save to the cwd folder.

    datafile_path = os.path.join(cwd, 'obj.data')

    datafile = open(datafile_path, "w")

    datafile.write("classes = " + str(classes) + "\n")

    datafile.write("train = " + empty_file + "\n")

    datafile.write("valid = " + empty_file + "\n")

    datafile.write("names = " + obj_names_path + "\n")
    
    datafile.write("backup = " + cwd)

    datafile.close()

    # Now we execute the Ultralytics detect.py script with our arguments to run the detection on our data.

    print("Running detection script on the input data. This could take some time...")

    check_output(['python detect.py --cfg ' + cfg_path + ' --weights ' + weights_path + ' --data-cfg ' + datafile_path + ' --images ' + input_path + ' --conf-thres ' + str(conf_thres)], shell = True)

    print("Done.")

    # Open output folder for classified images.

    print("Opening output classified image folder...")

    output_path = args['output']

    if output_path is None:
        raise NameError("No output image folder path, use the --output argument to supply the path. Exiting...")
    else:
        print("Done.")

    # Create the output folder if it doesn't exist

    if not os.path.exists(os.path.join(cwd, output_path)):

        os.makedirs(os.path.join(cwd, output_path))

    os.chdir(cwd)

    check_output(['cp ' + os.path.join(ultra_path, 'output', '*') + ' ' + output_path], shell = True)

    os.chdir(output_path)

    make_video = args['make_video']

    if make_video == True:

        print("Using ffdshow to produce a video from the classified frames (use on thermal video frames)....")

        framerate = args['video_framerate']

        if framerate is None:
            framerate = 30
            print("No video framerate supplied, use the --video_framerate argument to enter one. Defaulting to 30 fps... Done.")
        else:
            print("Video framerate of " + str(framerate) + " fps supplied... Done.")

        example_file = os.listdir(os.path.join(cwd, output_path))[0]

        example_filename, filetype = os.path.splitext(example_file)

        check_output(['ffmpeg -r ' + str(framerate) + ' -i frame_%6d' + filetype + ' -vcodec libx264 -pix_fmt yuv420p -crf 24 -an output.mp4'], shell = True)

        check_output(['mv output.mp4 ' + cwd], shell = True)


# Depending on the chosen mode, call the appropriate function.

if mode == "train":

    astroeco_ml_train(args, cwd, darknet_path)

elif mode == "eval":

    astroeco_ml_eval()

elif mode == "detect":

    astroeco_ml_detect(args, cwd, ultra_path)

else:

    raise NameError("'" + mode + "' is not a valid mode. You managed to avoid the first catch, I am impressed. Exiting...")

print("Operation complete.")
