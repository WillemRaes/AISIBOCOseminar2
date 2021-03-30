
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import threading
import socket
import time
import cv2
import msgpack
import msgpack_numpy as m
m.patch()

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """


COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
                               (0, 2),
                               (1, 3),
                               (2, 4),
                               (0, 5),
                               (0, 6),
                               (5, 7),
                               (7, 9),
                               (6, 8),
                               (8, 10),
                               (5, 6),
                               (5, 11),
                               (6, 12),
                               (11, 12),
                               (11, 13),
                               (13, 15),
                               (12, 14),
                               (14, 16)]

# Create a simple video logger
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("objectdetectionvideo" + str(time.time()) + "demopart2.avi", fourcc, 15, (512, 512))

# Path to the labels for objectdetection which in this case is mscoco
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
label_id_offset = 0
# Create a simple network socket to receive inference result and bounding boxes
HOST = ''           # all ipv4 addresses on this device
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP/IP
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()

print("Connected")
while True:
    data = bytes()
    bootstrap = None
    try:
        bootstrap = conn.recv(2)
        if bootstrap == b'AA':
            # hard coded packet length put length in first 4 bytes of packet instead
            while len(data) < 789126:
                data += conn.recv(4096)
            bootstrap = None
    except Exception as e:
        print(e)
    try:
        if data:
            data_deserialized = msgpack.unpackb(data)
            image_np_with_detections = data_deserialized[0].copy()
            result = data_deserialized[1]

            # Use keypoints if available in detections
            keypoints, keypoint_scores = None, None
            if 'detection_keypoints' in result:
                keypoints = result['detection_keypoints'][0]
                keypoint_scores = result['detection_keypoint_scores'][0]

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections[0],
                result['detection_boxes'][0],
                (result['detection_classes'][0] + label_id_offset).astype(int),
                result['detection_scores'][0],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)
            out.write(image_np_with_detections[0])
            cv2.imshow("With boxes", image_np_with_detections[0])
            cv2.waitKey(1)
            # plt.savefig("beach_obje.png")
    except Exception as e:
        print(e)


out and out.release()
cv2.destroyAllWindows()
