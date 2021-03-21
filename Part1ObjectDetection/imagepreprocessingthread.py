# import required libraries
import cv2

import threading
import time
import numpy as np
import struct
import logging
from Part1ObjectDetection.Config import Config


class ImagePreprocessingThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, image_size=(512, 512),
                 args=(), kwargs=None, verbose=None):
        super(ImagePreprocessingThread, self).__init__()
        self.target = target
        self.name = name
        self.image_size = image_size

        return

    def run(self):
        while True:
            try:
                if not Config.im_queue.empty():
                    im = Config.im_queue.get()

                    if im.shape == self.image_size:
                        Config.processed_im_queue.put(im_reshaped)

                    elif im.shape[0] % self.image_size[0] == 0 and im.shape[1] % self.image_size[1] == 0:
                        im_reshaped = cv2.resize(im, self.image_size)
                        # im.show()
                        Config.processed_im_queue.put(im_reshaped)
                    else:
                        # todo : add pixel overlap antialiasing fix
                        im_reshaped = cv2.resize(im, self.image_size)
                        Config.processed_im_queue.put(im_reshaped)

                    time.sleep(Config.approx_update_rate)

            except Exception as e:
                logging.debug(str(e))
                pass



        return


