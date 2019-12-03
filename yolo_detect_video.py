import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolo_models import YoloV3
from yolo_dataset import transform_images
from yolo_utils import draw_outputs

classes_file         = './data/coco.names'           #path to classes file
weights         = './checkpoints/yolov3.tf'     #path to weights file
tiny            = False                         # yolov3 or yolov3-tiny
fsize           =  416                          # resize images to
video_output    = './data/video.mp4'            #path to video file or number for webcam
output          = True                          #path to output video
output_format   = 'XVID'                        #codec used in VideoWriter when saving video to file
num_classes     =  80                           #number of classes in the model


def video_stream_detector():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo.load_weights(weights)
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


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
