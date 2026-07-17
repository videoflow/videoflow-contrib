'''
Detects and tracks faces in a video, then blurs them and writes the result to
blurred_video.mp4. The input video path is read from the ``VF_INPUT_VIDEO``
environment variable so the graph module can be imported by ``videoflow deploy``
without command-line args:

    VF_INPUT_VIDEO=/path/to/video.mp4 python face_obfuscation.py

Deploy to Kubernetes:

    videoflow deploy face_obfuscation.py:build_flow --nats nats://nats:4222 --image <your-image>
'''
import os

import numpy as np
import cv2

import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame

class BoundingboxObfuscator(videoflow.core.node.ProcessorNode):
    def __init__(self, nb_tasks = 1, **kwargs):
        super(BoundingboxObfuscator, self).__init__(nb_tasks = nb_tasks, **kwargs)

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

def build_flow():
    from videoflow_contrib.detector_tf import TensorflowObjectDetector
    from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker
    video_filepath = os.environ['VF_INPUT_VIDEO']
    reader = VideofileReader(video_filepath, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    faces = TensorflowObjectDetector(num_classes = 1,
        architecture = 'ssd-mobilenetv2',
        dataset = 'faces',
        min_score_threshold = 0.2,
        name = 'detector')(frame)
    tracked_faces = KalmanFilterBoundingBoxTracker(max_age = 12, min_hits = 0, name = 'tracker')(faces)
    blurred_faces = BoundingboxObfuscator(name = 'obfuscator')(frame, faces, tracked_faces)
    writer = VideofileWriter('blurred_video.mp4', codec = 'avc1', name = 'writer')(blurred_faces)
    return Flow([writer], flow_type = BATCH)

if __name__ == '__main__':
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()
