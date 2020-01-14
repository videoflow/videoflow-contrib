import sys

import numpy as np

import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
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

    def process(self, im: np.array, bounding_boxes: np.array):
        pass


def obfuscate_faces(video_filepath):
    reader = VideofileReader(video_filepath)
    frame = FrameIndexSplitter()(reader)
    faces = TensorflowObjectDetector(num_classes = 1, 
        path_to_pb_file = '/Users/dearj019/Documents/workspace/videoflow-contrib/releases/detector_tf/face.pb',
        min_score_threshold = 0.2)(frame)
    annotations = BoundingBoxAnnotator(class_labels_path = '/Users/dearj019/Documents/workspace/videoflow-contrib/releases/detector_tf/face_label_map.pbtxt')(frame, faces)
    writer = VideofileWriter('blurred_video.mp4', codec = 'avc1')(annotations)
    fl = flow.Flow([reader], [writer], flow_type = BATCH)
    fl.run()
    fl.join()

if __name__ == '__main__':
    video_file = sys.argv[1]
    obfuscate_faces(video_file)
