#!"C:\Python36\python.exe"

#test application of the object identifier part of the AI
# pipeline in an attempt to make a ... killbot I guess.
# of course half of this code is ripped straight from stackoverflow 
# or the documentation but then again... at first, what learning project isn't?
###########################################
# From "research" directory:
#   * protoc --python_out=. object_detection\protos\*.proto
#   * run with "py -3.6 ./script.py"
#   * we are NOT installing the libs, we are using it as a relative path module!
#   * this script goes into your "AI_BRAIN" directory, i.e. top level dir
#   * this let us package it easily and keep updates seperate.
#
###########################################
import os
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.python.client import device_lib
import _thread
from PIL import ImageGrab
import numpy as np
import cv2
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import colorama
from colorama import Fore, Back, Style
#start term color operation
#stops warnings about AVX2 support... we  usin' a GPU babeh'
#how manyy cores we are allowing this AI to use for object detection
#just some info for logging and development purposes
colorama.init()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_PARALLEL_EXEC_UNITS            = 4      # 0 is automatic set by tensorflow
GPU_NUM                            = 0

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name  = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model  = tf.keras.models.load_model(model_name)

print(detection_model.inputs)
print(Fore.CYAN + "DEVICE LIST:" + Style.RESET_ALL)
print(device_lib.list_local_devices())
print("DEVICE LIST (GRAPHICAL):" + Style.RESET_ALL)
# this is how you configure tensorflow to use individual cores, have to
# label them as individuals... im going to limit myself to 
# HALF my CPU please thank you.
# Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
# when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings 
config = tf.compat.v1.ConfigProto(device_count={"CPU": NUM_PARALLEL_EXEC_UNITS},
            allow_soft_placement=True,
            inter_op_parallelism_threads=2,
            intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS)
sess   = tf.compat.v1.Session(config=config)
gpus   = tf.config.experimental.list_physical_devices('GPU')

# The following example splits the GPU into 2 virtual devices with #Megabytes memory
# and locks the program to only one physical device
if assert len(gpus) > 0:
    print(Fore.RED + "No GPUs found" + Style.RESET_ALL)
else:
    # sets single physical device
    tf.config.experimental.set_visible_devices(gpus[GPU_NUM], 'GPU')
    # sets virtual devices
    tf.config.experimental.set_virtual_device_configuration(
    gpus[GPU_NUM],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus),Fore.RED +  "Physical GPUs," + Style.RESET_ALL, len(logical_gpus),Fore.MAGENTA +  "Logical GPU's" + Style.RESET_ALL)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(Fore.RED + Back.WHITE + e + Style.RESET_ALL)


# THIS is how you use those cores individually.
# timer added for debug and demonstration purposes
def cpu_compute(core_num):
    start = timer()
    with tf.device("/cpu:" + core_num):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    end = timer()
    print(c)
    print(end - start)

  # ...
#written for single gpu systems, change GPU_NUM for more
# timer added for debug and demonstration purposes
def gpu_compute(GPU_NUM):
    with tf.device("/gpu:" + GPU_NUM)
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    end = timer()
    print(c)
    print(end - start)

def screencapture():
    img = ImageGrab.grab(bbox=(100,10,400,780)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

_thread.start_new_thread ( screencapture, args[, kwargs] )