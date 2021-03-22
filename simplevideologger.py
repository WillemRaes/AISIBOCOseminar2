"""
This script loads a pre trained SOTA model of resnetv2 architecture with image net weights
The model in this form can be used for object detection without bounding boxes

"""

import numpy as np
import tensorflow as tf
from PIL import Image as Image
from six import BytesIO
from six.moves.urllib.request import urlopen
import cv2
IMAGE_SIZE = (224, 224)
from Part1ObjectDetection.Config import Config
from Part1ObjectDetection.rtspstreamreaderthread import RTSPStreamCaptureThread
from Part1ObjectDetection.imagepreprocessingthread import ImagePreprocessingThread
from Part1ObjectDetection.resnetLabels import resnetdict

# Be careful not to use the same queues from Config for both logging and inference on same machine
# Safer way is to pass the queues as parameters to the threads rather than using global variables from Config
c = ImagePreprocessingThread(name='Imagepreprocessing', image_size=(512, 512))
c.start()
p = RTSPStreamCaptureThread(name='RTSPreader', host='http://admin:admin@192.168.1.156:8080/video')
p.start()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("demopart1_attempt4.avi", fourcc, 25, (512, 512))


while True:
    try:
        if not Config.processed_im_queue.empty():
            im = Config.processed_im_queue.get()

            image_np = im
            print(image_np.shape)
            image_np = np.array(im)[None, :, :, :]
            # Flip horizontally
            print(image_np.shape)
            print(image_np[0].shape)
            cv2.imshow('image', image_np[0])
            cv2.waitKey(1)
            out.write(image_np[0])
            # cv2.imshow("With boxes", image_np_with_detections[0])
            # cv2.waitKey(1)
            # plt.savefig("beach_obje.png")
    except Exception as e:
        print(e)
        out and out.release()
        cv2.destroyAllWindows()


