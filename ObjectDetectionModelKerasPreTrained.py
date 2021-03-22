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


# Load a SoTa neural network model backbone for object detection
model = tf.keras.applications.ResNet50V2(
    include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
# In this demo we will not do any fine tuning or transfer learning so we freeze the model parameters immediately
model.trainable = False

# We use a stock image to test the model
image_path = "C:\\werk\\Tensorflow\\models\\research\\object_detection\\test_images\\image1.jpg"
image_np = load_image_into_numpy_array(image_path)
print(image_np.shape)

# Library utility function to preprocess an image for inference
# image_np = tf.keras.applications.resnet_v2.preprocess_input(image_np)

# Manual preprocessing on test image for inference
image_np = image_np / 127.5 - 1

# Run inference on the test image
res = model.predict(image_np)

# Get the Top 5 predictions
print(tf.keras.applications.resnet_v2.decode_predictions(
    res, top=5
))

# Save the model to custom Tensorflow saved model format
model.save('saved_models/objectResnetPretrained')

# saved_model_dir = "./saved_models/objectResnetPretrained"
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
