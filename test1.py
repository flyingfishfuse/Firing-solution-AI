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
model                              = tf.keras.models
#ENABLE DEBUGGERING!
tf.debugging.set_log_device_placement(True)

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
        print(Fore.MAGENTA + Back.WHITE + "Loading Dataset!" + Style.RESET_ALL)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    end = timer()
    print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)

def gpu_compute_timed2(gpunum):
    with tf.device(gpunum):
        start = timer()
        print(Fore.MAGENTA + Back.WHITE + "Loading Dataset!" + Style.RESET_ALL)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        end = timer()
        print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)
        print(Fore.MAGENTA + Back.WHITE + "Compiling Model!" + Style.RESET_ALL)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            #
            # OR you could use :
            #   import tensorflow.keras.layers as pile_o_layers
            #   model = tf.keras.models.Sequential()
            #   model.add(pile_o_layers.Dense(128, activation='relu'))
            #
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        end = timer()
        print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)
        start = timer()
        print(Fore.MAGENTA + Back.WHITE + "Fitting Model!" + Style.RESET_ALL)
        model.fit(x_train, y_train, epochs=5)
        end = timer()
        print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)
        start = timer()
        print(Fore.MAGENTA + Back.WHITE + "Evaluating Model!" + Style.RESET_ALL)
        model.evaluate(x_test,  y_test, verbose=2)
        end = timer()
        print(Fore.MAGENTA + str(end - start)  + Style.RESET_ALL)
    return model
gpu_compute_timed1(GPU_NUM)
asdf = gpu_compute_timed2(GPU_NUM)
# Display the model's architecture
asdf.summary()
