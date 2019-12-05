#!"C:\Python36\python.exe"

#test application of the object identifier part of the AI
# pipeline in an attempt to make a ... killbot I guess.
# of course half of this code is ripped straight from stackoverflow 
# or the documentation but then again... at first, what learning project isn't?
# 
#   * CUDA 10
#   * cuDNN (look this version up , be better than those schlups doing shitty documentation)
#   * python 3.6
#   * tensorflow 2.0
#   * 
###########################################
# From "research" directory:
#   * protoc --python_out=. object_detection\protos\*.proto
#   * run with "py -3.6 ./script.py"
#   * put yolov3 in the top level directory
#   * this script goes into your "AI_BRAIN" directory, i.e. top level dir
#   * 
#
###########################################
import os
import cv2
import time
import zipfile
import colorama
import numpy as np
from PIL import Image
import tensorflow as tf
from io import StringIO
from PIL import ImageGrab
from absl.flags import FLAGS
from IPython.display import display
from collections import defaultdict
from matplotlib import pyplot as plt
from absl import app, flags, logging
from yolo_models import YoloV3
from colorama import Fore, Back, Style
from timeit import default_timer as timer
from yolo_utils import draw_outputs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.client import device_lib
from yolo_dataset import transform_images
from yolo_models import YoloV3, YoloV3Tiny
from yolo_utils import load_darknet_weights
from object_detection.utils import label_map_util
from tensorflow.compat.v1 import InteractiveSession
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolo_models import (
    YoloV3, 
    YoloLoss,
    yolo_anchors, 
    yolo_anchor_masks,
)
from yolo_utils import freeze_all
import yolo_dataset as yolo_dataset


#start term color operation
#stops warnings about AVX2 support... we  usin' a GPU babeh'
#how manyy cores we are allowing this AI to use for object detection
#just some info for logging and development purposes
colorama.init()

YOLO_STARTUP_PARAMS                = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_PARALLEL_EXEC_UNITS            = 4      # 0 is automatic set by tensorflow
GPU_NUM                            = 0      # 0 is automatic set by tensorflow
NUM_CLASSES                        = 90
GPU_MEMORY_LIMIT_PER_GPU           = 1024
GPU_FRACTION_LIMIT_BOOL            = False
GPU_FRACTION_LIMIT                 = .25
# List of the strings that is used to add correct label for each box.
MODEL_NAME               = 'ssd_inception_v2_coco_2018_1_28'
#PATH_TO_MODEL            = 'models/' + MODEL_NAME
PATH_TO_LABELS           = 'mscoco_label_map.pbtxt'
LABEL_MAP                = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES               = label_map_util.convert_label_map_to_categories(LABEL_MAP, max_num_classes=NUM_CLASSES, use_display_name=True)
CATEGORY_INDEX           = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
DETECTION_MODEL          = tf.keras.models.load_model('./')
PATH_TO_TEST_IMAGES_DIR  = ''
TEST_IMAGE_PATHS         = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'meme.jpg')]

path_to_classes          = './data/coco.names'
path_to_weights          = './data/yolov3.weights'  #path to weights file
image_resize_size        = 416
INPUT_image              = './data/girl.png'
output                   = '.output.jpg'
num_classes              = 80
yolo_iou_threshold       = 0.5
yolo_score_threshold     = 0.5

yolo_dataset        = None                              #path to yolo_dataset
val_dataset         = None                              #path to validation yolo_dataset

# tiny              = False                         #yolov3 or yolov3-tiny

classes             = './data/coco.names'           #path to classes file
epochs              = 2                             #number of epochs
batch_size          = 8                             #batch size
learning_rate       = 1e-3                          #learning rate
num_classes         = 80                            #number of classes in the model
classes_file        = './data/coco.names'           #path to classes file
weights             = './checkpoints/yolov3.tf'     #path to weights file
tiny                = False                         # yolov3 or yolov3-tiny
fsize               =  416                          # resize images to
video_output        = './data/video.mp4'            #path to video file or number for webcam
output              = True                          #path to output video
output_format       = 'XVID'                        #codec used in VideoWriter when saving video to file

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')

flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')

print(DETECTION_MODEL)

# this is how you configure tensorflow to use individual cores, have to label them as individuals...   
# im going to limit myself to HALF my CPU please thank you.
# Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
# when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings 
#   * we are using 4 cores for testing, set the number at the top of the file
def setup_processor_configuration(allow_growth=True):
    config = tf.compat.v1.ConfigProto(device_count={"CPU": NUM_PARALLEL_EXEC_UNITS},
            allow_soft_placement=True,
            inter_op_parallelism_threads=2,
            intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS)
    gpus   = tf.config.experimental.list_physical_devices('GPU')
    # The following example splits the GPU into 2 virtual devices with #Megabytes memory
    # and locks the program to only one physical device
    if len(gpus) > 0:
        print(Fore.RED + "No GPUs found" + Style.RESET_ALL)
    
    else:
        # sets single physical device
        tf.config.experimental.set_visible_devices(gpus[GPU_NUM], 'GPU')
        # sets virtual devices
        if GPU_FRACTION_LIMIT_BOOL == True :
            config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION_LIMIT
        else:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[GPU_NUM],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_PER_GPU),
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_PER_GPU)])
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, allow_growth)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus),Fore.RED +  "Physical GPUs," + Style.RESET_ALL, len(logical_gpus),Fore.MAGENTA +  "Logical GPU's" + Style.RESET_ALL)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(Fore.RED + Back.WHITE + e + Style.RESET_ALL)

def startup_yolo(YOLO_STARTUP_PARAMS):
    yolo = YoloV3( classes= num_classes)
    yolo.summary()
    logging.info('model created')
    load_darknet_weights(yolo, path_to_weights)
    logging.info('weights loaded')
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')
    yolo.save_weights(output)
    logging.info('weights saved')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = YOLOV3_LAYER_LIST
    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]
            logging.info("{}/{} {}".format(sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]
            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))
    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def screencapture():
    img = ImageGrab.grab(bbox=(100,10,400,780)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_and_draw():
    yolo = YoloV3(classes= num_classes)
    yolo.load_weights(FLAGS.path_to_weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(path_to_classes).readlines()]
    logging.info('classes loaded')
    img = tf.image.decode_image(open(INPUT_image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, fsize)
    t1 = time.time()
    boxes, scores, num_classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))
    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[
            int(classes[0][i])],
            np.array(scores[0][i]),
            np.array(boxes[0][i])
            )
        )
    img = cv2.imread(FLAGS.image)
    img = draw_outputs(img, (boxes, scores, num_classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(video_output))


def video_stream_detector():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo.load_weights(path_to_weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(classes_file).readlines()]
    logging.info('classes loaded')
    times = []
    try:
        vid = cv2.VideoCapture(int(video_output))
    except:
        vid = cv2.VideoCapture(video_output)
    out = None                                                                                   
    if output == True:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        img_in = tf.expand_dims(img, 0)
        img_in = transform_images(img_in, FLAGS.size)
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


 def start_capture_threads():   
    _thread.start_new_thread ( screencapture, args[, kwargs] )


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
