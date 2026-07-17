'''
Blurs detected faces in a video and writes the result to blurred_video.mp4.

The input video path is read from the ``VF_INPUT_VIDEO`` environment variable so
the graph module can be imported by ``videoflow deploy`` without command-line args:

    VF_INPUT_VIDEO=/path/to/video.mp4 python examples/face_obfuscation.py

Deploy to Kubernetes:

    videoflow deploy examples/face_obfuscation.py:build_flow --nats nats://nats:4222 --image <your-image>
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
    '''Drops the frame index from a (index, frame) producer tuple, keeping the frame.'''
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame

class BoundingboxObfuscator(videoflow.core.node.ProcessorNode):
    def __init__(self, nb_tasks = 1, **kwargs):
        super(BoundingboxObfuscator, self).__init__(nb_tasks = nb_tasks, **kwargs)

    def process(self, im: np.array, bounding_boxes: np.array):
        '''
        - Arguments:
            - im: np.array of shape (h, w, 3)
            - bounding_boxes: np.array of shape (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
        '''
        result_image = im.copy()
        for box in bounding_boxes:
            ymin, xmin, ymax, xmax, _, _ = box.astype(int)
            sub_face = im[ymin : ymax, xmin : xmax, :]
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            result_image[ymin : ymax, xmin : xmax, :] = sub_face
        return result_image

def build_flow():
    from videoflow_contrib.detector_tf import TensorflowObjectDetector
    video_filepath = os.environ['VF_INPUT_VIDEO']
    reader = VideofileReader(video_filepath, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    faces = TensorflowObjectDetector(num_classes = 1,
        architecture = 'ssd-mobilenetv2',
        dataset = 'faces',
        min_score_threshold = 0.2,
        name = 'detector')(frame)
    blurred_faces = BoundingboxObfuscator(name = 'obfuscator')(frame, faces)
    writer = VideofileWriter('blurred_video.mp4', codec = 'avc1', name = 'writer')(blurred_faces)
    return Flow([writer], flow_type = BATCH)

if __name__ == '__main__':
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()
