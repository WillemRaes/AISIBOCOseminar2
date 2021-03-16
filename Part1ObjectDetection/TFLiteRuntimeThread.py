"""
This thread loads the ONNX runtime for inference with the saved model
The standard tf2onnx cmd tool can be used for this
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx
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
from Config import Config
from resnetLabels import resnetdict
import tensorflow as tf


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
        # interpreter = tf.lite.Interpreter(model_path=self.modelName)
        # interpreter.allocate_tensors()
        # counter = 0
        # # Get input and output tensors.
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # print(output_details)
        print("here")
        model = tf.keras.models.load_model("./ssd_mobilenet_v2_2")
        print("model loaded")
        while True:
            try:
                if not Config.processed_im_queue.empty():
                    im = Config.processed_im_queue.get()
                    # im.show()
                    im_temp_fix = np.array(im)[None, :, :, :]

                    output_data = model.predict(im_temp_fix)
                    print(tf.keras.applications.resnet_v2.decode_predictions(
                        output_data, top=5))
                    # top_pred = np.where(output_data.squeeze() > 0.5)
                    #
                    # for arr in top_pred:
                    #     for val in arr:
                    #         label = resnetdict.resnet_labels.get(int(val))
                    #         print("Object: " + label + str(output_data.squeeze()[val]))
                    #
                    counter += 1
                    if counter % 100 == 0:
                        time_diff = time.time() - start_time
                        print("Frames per second: ", int(counter / time_diff))
                        start_time = time.time()
                        counter = 0
                    time.sleep(0.001)

            except Exception as e:
                logging.debug(str(e))
                pass



        return

