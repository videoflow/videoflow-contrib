'''
Runs a Detectron2 human-pose estimator over a sample video and writes an annotated
copy to pose.avi.

Local run (needs a NATS server):

    python examples/humanpose_example.py

Deploy to Kubernetes:

    videoflow deploy examples/humanpose_example.py:build_flow --nats nats://nats:4222 --image <your-image>
'''
import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow-contrib/releases/download/example_videos/"
VIDEO_NAME = 'people_walking.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME


class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    '''Drops the frame index from a (index, frame) producer tuple, keeping the frame.'''
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame


class KeypointsExtractor(videoflow.core.node.ProcessorNode):
    '''Keeps only the keypoints from the (keypoints, bounding_boxes) detector output.'''
    def __init__(self, **kwargs):
        super(KeypointsExtractor, self).__init__(**kwargs)

    def process(self, data):
        keypoints, bounding_boxes = data
        return keypoints


def build_flow():
    from videoflow_contrib.detectron2 import Detectron2HumanPose, HumanPoseAnnotator
    video_filepath = get_file(VIDEO_NAME, URL_VIDEO)
    reader = VideofileReader(video_filepath, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    results = Detectron2HumanPose(architecture = "R50_FPN_3x", device_type = "cpu", name = 'pose')(frame)
    keypoints = KeypointsExtractor(name = 'keypoints')(results)
    annotated_frame = HumanPoseAnnotator(name = 'annotator')(frame, keypoints)
    writer = VideofileWriter("pose.avi", name = 'writer')(annotated_frame)
    return Flow([writer], flow_type = BATCH)

if __name__ == '__main__':
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()
