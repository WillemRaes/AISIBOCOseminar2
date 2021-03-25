""" ____  ____      _    __  __  ____ ___
|  _ \|  _ \    / \  |  \/  |/ ___/ _ \
| | | | |_) |  / _ \ | |\/| | |  | | | |
| |_| |  _ <  / ___ \| |  | | |__| |_| |
|____/|_| \_\/_/   \_\_|  |_|\____\___/
                          research group
                            dramco.be/

 KU Leuven - Technology Campus Gent,
 Gebroeders De Smetstraat 1,
 B-9000 Gent, Belgium

     Author: Willem Raes
     Version: 1.0
"""

import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen
import tensorflow as tf
from Part1ObjectDetection.rtspstreamreaderthread import RTSPStreamCaptureThread
from Part1ObjectDetection.imagepreprocessingthread import ImagePreprocessingThread
from Part1ObjectDetection.Config import Config
import socket
import msgpack
import msgpack_numpy as m
m.patch()

c = ImagePreprocessingThread(name='Imagepreprocessing', image_size=(512, 512))
c.start()
p = RTSPStreamCaptureThread(name='RTSPreader', host='http://admin:admin@192.168.1.156:8080/video')
p.start()


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    image = None
    if (path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
                               (0, 2),
                               (1, 3),
                               (2, 4),
                               (0, 5),
                               (0, 6),
                               (5, 7),
                               (7, 9),
                               (6, 8),
                               (8, 10),
                               (5, 6),
                               (5, 11),
                               (6, 12),
                               (11, 12),
                               (11, 13),
                               (13, 15),
                               (12, 14),
                               (14, 16)]

model = tf.saved_model.load("./saved_models/CenternetObjectDetectionBoxes")
print("Model Loaded")

# create minimal socket connection to server screen
# HOST = "127.0.0.1"
HOST = '192.168.1.135'  # The server's hostname or IP address
PORT = 65432  # The port used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("Connected")
detections=10

while True:
    try:
        if not Config.processed_im_queue.empty():
            im = Config.processed_im_queue.get()
            # selected_image = im
            flip_image_horizontally = False
            convert_image_to_grayscale = False

            # image_path = IMAGES_FOR_TEST[selected_image]
            image_np = im.reshape(1, 512, 512, 3)

            # Flip horizontally
            if flip_image_horizontally:
                image_np[0] = np.fliplr(image_np[0]).copy()

            # Convert image to grayscale
            if convert_image_to_grayscale:
                image_np[0] = np.tile(
                    np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            # running inference
            results = model(image_np)

            # different object detection models have additional results
            # all of them are explained in the documentation
            result = {key: value.numpy() for key, value in results.items()}
            print(result.keys())

            # filter out highest prediction scores
            result['detection_boxes'] = result['detection_boxes'][0: detections + 1]
            result['detection_classes'] = result['detection_classes'][0: detections + 1]
            result['detection_scores'] = result['detection_scores'][0: detections + 1]

            image_np_with_detections = image_np.copy()

            to_send = [image_np_with_detections, result]

            # Send inference result and bounding box to screen over network socket
            bin_buff = msgpack.packb(to_send)
            print(len(bin_buff))
            s.sendall(b'AA')
            s.sendall(msgpack.packb(to_send))

    except Exception as e:
        print(e)
