"""
Main file where threads are started

"""

import logging
logging.basicConfig(level=logging.DEBUG, format='(%(asctime)s %(threadName)-9s) %(message)s', filename="log.txt")
# from httpclientthread import HttpStreamCaptureThread
from Part1ObjectDetection.rtspstreamreaderthread import RTSPStreamCaptureThread
from Part1ObjectDetection.imagepreprocessingthread import ImagePreprocessingThread
from Part1ObjectDetection.TFRuntimeThread import TFLiteRuntimeThread
from Part1ObjectDetection.ResnetOnnxRuntimeThread import ResnetOnnxRuntimeInferenceThread


def main():

    c = ImagePreprocessingThread(name='Imagepreprocessing', image_size=(224, 224))
    c.start()
    logging.debug("Started Consumerthread")
    # p = RTSPStreamCaptureThread(name='RTSPreader', host='rtsp://admin:ditisveiliger@192.168.1.191:554/11')
    p = RTSPStreamCaptureThread(name='RTSPreader', host='http://admin:admin@192.168.1.156:8080/video')
    p.start()
    logging.debug("Started rtsp_client")
    # inf_thread = ResnetOnnxRuntimeInferenceThread(name="ONNXRuntimeObjectDetect", modelName="resnetpretrainedtest_trainablefalse.onnx")
    # inf_thread.start()

    inf_thread_tflite = TFLiteRuntimeThread(name="TFliteRuntime", modelName="kerasmodel.tflite")
    inf_thread_tflite.start()

if __name__ == '__main__':
    main()


