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
        image = image.resize(IMAGE_SIZE)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)  # removed prefix 1 from shape


model = tf.keras.applications.ResNet50V2(
    include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
model.trainable = False
print(model.summary())
image_path = "C:\\werk\\Tensorflow\\models\\research\\object_detection\\test_images\\image1.jpg"

image_np = load_image_into_numpy_array(image_path)
print(image_np.shape)
# image_np = tf.keras.applications.resnet_v2.preprocess_input(image_np)
image_np = image_np / 127.5 - 1
res = model.predict(image_np)
print(tf.keras.applications.resnet_v2.decode_predictions(
    res, top=5
))
# model.save('saved_models/objectResnetPretrained')
#
# saved_model_dir = "./saved_models/objectResnetPretrained"
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
run_debug = True
if run_debug:
    # Message queues are global for the moment better to pass them as argument
    c = ImagePreprocessingThread(name='Imagepreprocessing', image_size=(224, 224))
    c.start()
    p = RTSPStreamCaptureThread(name='RTSPreader', host='http://admin:admin@192.168.1.156:8080/video')
    p.start()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("testvideo3.avi", fourcc, 20, (224, 224))


    while True:
        try:
            if not Config.processed_im_queue.empty():
                im = Config.processed_im_queue.get()
                # selected_image = im
                flip_image_horizontally = False
                convert_image_to_grayscale = False

                # image_path = IMAGES_FOR_TEST[selected_image]
                image_np = im
                print(image_np.shape)
                image_np = np.array(im)[None, :, :, :]

                print(image_np.shape)
                print(image_np[0].shape)
                cv2.imshow('image', image_np[0])
                cv2.waitKey(1)


                image_np = image_np / 127.5 - 1
                # running inference
                results = model.predict(image_np)
                # Get class labels without using decode in tf.keras api (ONNX goal to remove dependencies)
                print(results.shape)
                top_pred = np.where(results.squeeze() > 0.5)

                for arr in top_pred:
                    for val in arr:
                        label = resnetdict.resnet_labels.get(int(val))
                        print("Object: " + label + str(results.squeeze()[val]))
                out.write(image_np[0])
                # cv2.imshow("With boxes", image_np_with_detections[0])
                # cv2.waitKey(1)
                # plt.savefig("beach_obje.png")
        except Exception as e:
            print(e)
            out and out.release()
            cv2.destroyAllWindows()


