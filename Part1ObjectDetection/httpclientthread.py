# import required libraries
from vidgear.gears import NetGear
import cv2
import http
import requests
from PIL import Image
from io import BytesIO
import threading
import time
import logging
import numpy as np
import struct
import logging
from Config import Config


class HttpStreamCaptureThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, host='http://192.168.137.153',
                 args=(), kwargs=None, verbose=None):
        super(HttpStreamCaptureThread, self).__init__()
        self.target = target
        self.name = name
        self.host = host

        return

    def run(self):
        images = []
        try:
            r = requests.get(self.host, timeout=5)

        except Exception as e:
            print(e)
            return

        for i in range(200):
            try:
                r2 = requests.get(self.host + '/capture', timeout=2)
                if r2.status_code == 200:

                    im = Image.open(BytesIO(r2.content))
                    if not Config.im_queue.full():
                        Config.im_queue.put(im)

                    # im.show()
                    time.sleep(Config.approx_update_rate)


            except Exception as e:
                print(e)
                pass


        return


