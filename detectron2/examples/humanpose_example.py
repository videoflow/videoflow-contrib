import sys

import videoflow
import videoflow.core.flow as flow
from videoflow.consumers import VideofileWriter
from videoflow.core.constants import BATCH
from videoflow.producers import VideofileReader
from videoflow_contrib.detectron2 import Detectron2HumanPose, HumanPoseAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow-contrib/releases/download/example_videos/"
VIDEO_NAME = 'people_walking.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME


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


def annotate_video(video_filepath):
    reader = VideofileReader(video_filepath)
    frame = FrameIndexSplitter()(reader)
    results = Detectron2HumanPose(architecture="R50_FPN_3x", device_type="cpu")(frame)
    keypoints = KeypointsExtractor()(results)
    annotated_frame = HumanPoseAnnotator()(frame, keypoints)
    writer = VideofileWriter("pose.avi")(annotated_frame)
    fl = flow.Flow([reader], [writer], flow_type=BATCH)
    fl.run()
    fl.join()

if __name__ == '__main__':
    video_filepath = get_file(
        VIDEO_NAME,
        URL_VIDEO
    )
    annotate_video(video_filepath)
