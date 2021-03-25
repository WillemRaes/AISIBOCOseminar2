"""
This thread loads the ONNX runtime for inference with the saved model
The standard tf2onnx cmd tool can be used for this
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx
This thread is specific for resnet architecture family

"""

import onnxruntime as rt
import threading
import time
import time
import logging
import numpy as np
import struct
import logging
from Part1ObjectDetection.Config import Config
from Part1ObjectDetection.resnetLabels import resnetdict


class ResnetOnnxRuntimeInferenceThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, modelName="Resnet.onnx", classifierThreshold=0.4, predictionBatchSize=5,
                 args=(), kwargs=None, verbose=None):
        super(ResnetOnnxRuntimeInferenceThread, self).__init__()
        self.predictionBatchSize = predictionBatchSize
        self.classifierThreshold = classifierThreshold
        self.target = target
        self.name = name
        self.modelName = modelName
        return

    def run(self):

        # Get the runtime up
        sess_options = rt.SessionOptions()
        sess = rt.InferenceSession(self.modelName)
        input_name = sess.get_inputs()[0].name
        print("input name", input_name)
        input_shape = sess.get_inputs()[0].shape
        print("input shape", input_shape)
        input_type = sess.get_inputs()[0].type
        print("input type", input_type)
        label_name = sess.get_outputs()[0].name
        print("output name", label_name)
        output_shape = sess.get_outputs()[0].shape
        print("output shape", output_shape)
        output_type = sess.get_outputs()[0].type
        print("output type", output_type)

        prediction_batch = []
        input_name = sess.get_inputs()[0].name
        start_time = time.time()
        counter = 0
        while True:
            try:
                if not Config.processed_im_queue.empty():
                    im = Config.processed_im_queue.get()

                    im_temp_fix = np.array(im)  # [None, :, :, :]  # only needed when 1 shot prediction

                    # Resnet specific preprocessing requires input in [-1 1]
                    # better to make this part of the model
                    im_temp_fix = im_temp_fix / 127.5 - 1

                    prediction_batch.append(im_temp_fix)
                    if len(prediction_batch) == self.predictionBatchSize:
                        # do prediction on a batch of images
                        pred_onx = sess.run([label_name], {input_name: np.array(prediction_batch, dtype=np.float32)})[0]

                        # Get class labels without using decode in tf.keras api (ONNX goal to remove dependencies)
                        for output in pred_onx:
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
                        time.sleep(0.001)

            except Exception as e:
                logging.debug(str(e))

        return

