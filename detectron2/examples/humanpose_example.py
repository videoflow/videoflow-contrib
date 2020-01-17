import sys

import numpy as np

import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detectron2 import Detectron2HumanPose, HumanPoseAnnotator

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

class KeypointsExtractor(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(KeypointsExtractor, self).__init__()
    
    def process(self, data):
        keypoints, bounding_boxes = data
        return keypoints

def annotate_video(video_filepath, model_filepath, config_filepath):
    reader = VideofileReader(video_filepath)
    frame = FrameIndexSplitter()(reader)
    results = Detectron2HumanPose(model_filepath, config_filepath, device_type = 'cpu')(frame)
    keypoints = KeypointsExtractor()(results)
    annotated_frame = HumanPoseAnnotator()(frame, keypoints)
    writer = VideofileWriter('pose.avi')(annotated_frame)
    fl = flow.Flow([reader], [], flow_type = BATCH)
    fl.run()
    fl.join()

if __name__ == '__main__':
    video_filepath = sys.argv[1]
    model_filepath = sys.argv[2]
    config_filepath = sys.argv[3]
    annotate_video(video_filepath, model_filepath, config_filepath)