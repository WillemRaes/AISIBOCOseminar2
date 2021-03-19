import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# model_handle = 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1'
# hub_model = hub.KerasLayer(model_handle)
# saved_model_dir = "./tensorflowstuff/saved_models/CenternetObjectDetectionBoxes"
#
# # model = tf.saved_model.load(saved_model)
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()
#
# with open('CenternetWithBoxes_quant.tflite', 'wb') as f:
#   f.write(tflite_quant_model)

# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4")
# ])

saved_model_dir = "./saved_models/objectResnetPretrained"

# model = tf.saved_model.load(saved_model)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open('Part1ObjectDetection/resnet_quant.tflite', 'wb') as f:
  f.write(tflite_quant_model)