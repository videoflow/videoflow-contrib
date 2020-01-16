import sys

import numpy as np
import cv2

import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

class BoundingboxObfuscator(videoflow.core.node.ProcessorNode):
    def __init__(self, nb_tasks = 1):
        super(BoundingboxObfuscator, self).__init__(nb_tasks = nb_tasks)

    def process(self, im: np.array, bounding_boxes: np.array, tracker_bounding_boxes: np.array):
        '''
        - Arguments:
            - im: np.array of shape (h, w, 3)
            - bounding_boxes: np.array of shape (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
            - tracker_bounding_boxes: np.array of shape (nb_boxes, [[ymin, xmin, ymax, xmax, tracker_id]])
        '''
        result_image = im.copy()

        # Increasing the boxes sizes
        bounding_boxes_to_use = np.concatenate((bounding_boxes[:,0:4], tracker_bounding_boxes[:,0:4]))
        bounding_boxes_to_use[:,0] -= (0.20 * (bounding_boxes_to_use[:,2] - bounding_boxes_to_use[:,0]))
        bounding_boxes_to_use[:,2] += (0.20 * (bounding_boxes_to_use[:,2] - bounding_boxes_to_use[:,0]))
        bounding_boxes_to_use[:,1] -= (0.20 * (bounding_boxes_to_use[:,3] - bounding_boxes_to_use[:,1]))
        bounding_boxes_to_use[:,3] += (0.20 * (bounding_boxes_to_use[:,3] - bounding_boxes_to_use[:,1]))
        bounding_boxes_to_use[:,[0,1]] = np.maximum(bounding_boxes_to_use[:,[0,1]], 0)
        bounding_boxes_to_use[:,2] = np.minimum(bounding_boxes_to_use[:,2], im.shape[0])
        bounding_boxes_to_use[:,3] = np.minimum(bounding_boxes_to_use[:,3], im.shape[1])
        bounding_boxes_to_use = bounding_boxes_to_use.astype(np.int32)

        for box in bounding_boxes_to_use:
            ymin, xmin, ymax, xmax = box
            sub_face = im[ymin : ymax, xmin : xmax, :]
            try:
                sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            except:
                # sub_face shape has 0 dimensions probably.
                pass
            result_image[ymin : ymax, xmin : xmax, :] = sub_face
        return result_image

def obfuscate_faces(video_filepath):
    reader = VideofileReader(video_filepath)
    frame = FrameIndexSplitter()(reader)
    faces = TensorflowObjectDetector(num_classes = 1, 
        architecture = 'ssd-mobilenetv2',
        dataset = 'faces',
        min_score_threshold = 0.2)(frame)
    tracked_faces = KalmanFilterBoundingBoxTracker(max_age = 12, min_hits = 0)(faces)
    blurred_faces = BoundingboxObfuscator()(frame, faces, tracked_faces)
    writer = VideofileWriter('blurred_video.mp4', codec = 'avc1')(blurred_faces)
    fl = flow.Flow([reader], [writer], flow_type = BATCH)
    fl.run()
    fl.join()

if __name__ == '__main__':
    video_file = sys.argv[1]
    obfuscate_faces(video_file)
