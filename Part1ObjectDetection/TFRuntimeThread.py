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
from Part1ObjectDetection.Config import Config
from Part1ObjectDetection.resnetLabels import resnetdict
import tensorflow as tf


class TFRuntimeThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, modelName="saved_model", classifierThreshold=0.4, predictionBatchSize=5,
                 args=(), kwargs=None, verbose=None):
        super(TFRuntimeThread, self).__init__()
        self.classifierThreshold = classifierThreshold
        self.predictionBatchSize = predictionBatchSize
        self.target = target
        self.name = name
        self.modelName = modelName

        return

    def run(self):

        counter = 0
        prediction_batch = []

        model = tf.keras.models.load_model(self.modelName)
        print("model loaded")
        start_time = time.time()
        while True:
            try:
                if not Config.processed_im_queue.empty():
                    im = Config.processed_im_queue.get()
                    # im.show()
                    im_norm = im / 127.5 - 1

                    im_temp_fix = np.array(im_norm)  # [None, :, :, :] # needed for single image inference
                    prediction_batch.append(im_temp_fix)
                    if len(prediction_batch) == self.predictionBatchSize:

                        output_data = model.predict(np.array(prediction_batch))
                        for output in output_data:
                            # print(tf.keras.applications.resnet_v2.decode_predictions(
                            #     output, top=5))
                            top_pred = np.where(output.squeeze() > self.classifierThreshold)
                            for arr in top_pred:
                                for val in arr:
                                    label = resnetdict.resnet_labels.get(int(val))
                                    print("Object: " + label + str(output.squeeze()[val]))

                        prediction_batch = []
                        counter += self.predictionBatchSize
                        if counter % 50 == 0:
                            time_diff = time.time() - start_time
                            print("Frames per second: ", int(counter / time_diff))
                            start_time = time.time()
                            counter = 0


            except Exception as e:
                logging.debug(str(e))




        return

