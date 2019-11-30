from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.client import device_lib
import colorama
from colorama import Fore, Back, Style
from timeit import default_timer as timer

colorama.init()

#gotta be the same
os.environ["CUDA_VISIBLE_DEVICES"]="0"
GPU_NUM                            = '/GPU:0'      
GPU_DESIGNATION                    = 0
ALLOW_MEMORY_GROWTH                = True
GPU_MEMORY_LIMIT_PER_GPU           = 1024
GPU_FRACTION_LIMIT                 = .25
NUM_PARALLEL_EXEC_UNITS            = 1
NUM_PARALLEL_INTEROP               = 1
print(tf.version.VERSION)


config = tf.compat.v1.ConfigProto(device_count={"CPU": 0},
            allow_soft_placement=True,
            inter_op_parallelism_threads=NUM_PARALLEL_INTEROP,
            intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS)
gpus   = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[GPU_DESIGNATION], 'GPU')
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION_LIMIT
tf.config.experimental.set_memory_growth(gpus[GPU_DESIGNATION], ALLOW_MEMORY_GROWTH)
print(len(gpus),Fore.RED +  "Physical GPUs," + Style.RESET_ALL)

def gpu_compute_timed1(gpunum):
    start = timer()
    with tf.device(gpunum):
        print(Fore.MAGENTA + Back.WHITE + "Training Model!" + Style.RESET_ALL)
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_labels = train_labels[:1000]
        test_labels = test_labels[:1000]
        train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
        test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    end = timer()
    print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)

def gpu_compute_timed2(gpunum):
    start = timer()
    with tf.device(gpunum):
        print(Fore.MAGENTA + Back.WHITE + "Compiling Model!" + Style.RESET_ALL)
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')])
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    end = timer()
    print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)
    return model

gpu_compute_timed1(GPU_NUM)
# Create a basic model instance
model = gpu_compute_timed2(GPU_NUM)

# Display the model's architecture
model.summary()
