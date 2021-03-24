"""
This thread loads the ONNX runtime for inference with the saved model
The standard tf2onnx cmd tool can be used for this
python -m tf2onnx.convert --saved model /Path/to/savedmodel --output resnet50.onnx
This thread is specific for resnet architecture family
"""
import requests
from PIL import Image
from io import BytesIO
import threading
import time
import time
import logging
import numpy as np
import struct
import logging
from Part1ObjectDetection.Config import Config
from Part1ObjectDetection.resnetLabels import resnetdict
import tflite_runtime.interpreter as tflite


class TFLiteRuntimeThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, modelName="Resnet.tflite",
                 args=(), kwargs=None, verbose=None):
        super(TFLiteRuntimeThread, self).__init__()
        self.target = target
        self.name = name
        self.modelName = modelName
        return

    def run(self):

        # Load the TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=self.modelName)
        interpreter.allocate_tensors()
        counter = 0
        # # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(output_details)

        # model = tf.keras.models.load_model("./objectResnetPretrained")

        while True:
            try:
                if not Config.processed_im_queue.empty():
                    im = Config.processed_im_queue.get()
                    # im.show()
                    im_norm = im / 127.5 - 1

                    im_temp_fix = np.array(im_norm, dtype=np.float32)[None, :, :, :]

                    interpreter.set_tensor(input_details[0]['index'], im_temp_fix)

                    interpreter.invoke()

                    # The function `get_tensor()` returns a copy of the tensor data.
                    # Use `tensor()` in order to get a pointer to the tensor.
                    output_data = interpreter.get_tensor(output_details[0]['index'])

                    top_pred = np.where(output_data.squeeze() > 0.5)

                    for arr in top_pred:
                        for val in arr:
                            label = resnetdict.resnet_labels.get(int(val))
                            print("Object: " + label + str(output_data.squeeze()[val]))

                    counter += 1
                    if counter % 100 == 0:
                        time_diff = time.time() - start_time
                        print("Frames per second: ", int(counter / time_diff))
                        start_time = time.time()
                        counter = 0
                    time.sleep(0.001)

            except Exception as e:
                logging.debug(str(e))




        return

