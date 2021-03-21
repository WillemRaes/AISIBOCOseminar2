import queue
import numpy as np


class ConfigPart2:

    im_queue = queue.Queue(maxsize=5)
    processed_im_queue = queue.Queue(maxsize=5)
    # image_size = (224, 224)
    # image_size = (512, 512)
    approx_update_rate = 0.001
    # labels = np.array(["cloudy", "sunny", "rainy", "foggy", "snowy"])
