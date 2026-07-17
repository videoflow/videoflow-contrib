'''
Runs a TensorFlow instance segmenter over a sample intersection video and writes an
annotated copy to output.avi.

Local run (needs a NATS server):

    python examples/segmenter.py

Deploy to Kubernetes:

    videoflow deploy examples/segmenter.py:build_flow --nats nats://nats:4222 --image <your-image>
'''
import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.annotators import SegmenterAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    '''Drops the frame index from a (index, frame) producer tuple, keeping the frame.'''
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame

def build_flow():
    from videoflow_contrib.segmentation_tf import TensorflowSegmenter
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"
    reader = VideofileReader(input_file, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    detector = TensorflowSegmenter(name = 'segmenter')(frame)
    annotator = SegmenterAnnotator(name = 'annotator')(frame, detector)
    writer = VideofileWriter(output_file, fps = 30, name = 'writer')(annotator)
    return Flow([writer], flow_type = BATCH)

if __name__ == "__main__":
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()
