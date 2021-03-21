"""
This thread loads the ONNX runtime for inference with the saved model
The standard tf2onnx cmd tool can be used for this
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx

"""

import onnxruntime as rt
import threading
import time
import logging
import numpy as np
import struct
import logging
from Config import Config


class OnnxRuntimeInferenceThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, modelName="weatherCNN.onnx",
                 args=(), kwargs=None, verbose=None):
        super(OnnxRuntimeInferenceThread, self).__init__()
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

        input_name = sess.get_inputs()[0].name
        start_time = time.time()
        counter = 0
        while True:
            try:
                if not Config.processed_im_queue.empty():
                    im = Config.processed_im_queue.get()
                    # im.show()
                    im_temp_fix = np.array(im)[None, :, :, :]
                    pred_onx = sess.run([label_name], {input_name: np.array(im_temp_fix, dtype=np.float32)})[0]

                    print(pred_onx)
                    mask = pred_onx > 0.5
                    mask = mask.ravel()
                    print(Config.labels[mask])

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

