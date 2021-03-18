"""
This script loads a pre trained SOTA model of resnetv2 architecture with image net weights
The model in this form can be used for object detection without bounding boxes

"""

import numpy as np
import tensorflow as tf
from PIL import Image as Image
from six import BytesIO
from six.moves.urllib.request import urlopen

IMAGE_SIZE = (224, 224)


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
image_path = "C:\\werk\\Tensorflow\\models\\research\\object_detection\\test_images\\image1.jpg"

image_np = load_image_into_numpy_array(image_path)
print(image_np.shape)
# image_np = tf.keras.applications.resnet_v2.preprocess_input(image_np)
image_np = image_np / 127.5 - 1
res = model.predict(image_np)
print(tf.keras.applications.resnet_v2.decode_predictions(
    res, top=5
))
model.save('saved_models/objectResnetPretrained')

saved_model_dir = "./saved_models/objectResnetPretrained"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# # Save the model.
# with open('Part1ObjectDetection/kerasmodel.tflite', 'wb') as f:
#   f.write(tflite_model)


# tflite quantizer
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.inference_input_type = tf.uint8  # or tf.uint8
# # converter.inference_output_type = tf.uint8
# tflite_quant_model = converter.convert()
#
# with open('Part1ObjectDetection/ResnetPretrained_quant.tflite', 'wb') as f:
#   f.write(tflite_quant_model)
#
# # tflite_path = 'ResnetPretrained_quant.tflite'
# # onnx_path = 'ResnetPretrained_quant.onnx'
# #
# # tflite2onnx.convert(tflite_path, onnx_path)
#
# # Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="Part1ObjectDetection/ResnetPretrained_quant.tflite")
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(image_np, dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)