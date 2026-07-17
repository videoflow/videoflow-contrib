'''
Tracks people across frames of a sample video with the Tracktor tracker and writes an
annotated copy out. The output path is read from the ``VF_OUTPUT_FILE`` environment
variable (default output.avi) so the graph module can be imported by ``videoflow
deploy`` without command-line args:

    VF_OUTPUT_FILE=tracked.avi python examples/people_tracking.py

Deploy to Kubernetes:

    videoflow deploy examples/people_tracking.py:build_flow --nats nats://nats:4222 --image <your-image>
'''
import os

import numpy as np

import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.annotators import TrackerAnnotator
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

class TracksToAnnotator(videoflow.core.node.ProcessorNode):
    def __init__(self, **kwargs):
        super(TracksToAnnotator, self).__init__(**kwargs)

    def process(self, tracks):
        '''
        - Arguments:
            - tracks: np.array of shape (nb_tracks, [xmin, ymin, xmax, ymax, score, track_id])

        - Returns:
            - tracks_for_annotator: np.array of shape (nb_tracks, [ymin, xmin, ymax, xmax, track_id])
        '''
        try:
            to_return = np.array(tracks[:,[1,0,3,2,5]])
        except:
            to_return = tracks
        return to_return

def build_flow():
    from videoflow_contrib.tracktor import TracktorFromFrames
    output_file = os.environ.get('VF_OUTPUT_FILE', 'output.avi')
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    reader = VideofileReader(input_file, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    tracks = TracktorFromFrames(name = 'tracktor')(frame)
    tracks_to_annotator = TracksToAnnotator(name = 'tracks-to-annotator')(tracks)
    annotator = TrackerAnnotator(name = 'annotator')(frame, tracks_to_annotator)
    writer = VideofileWriter(output_file, fps = 30, name = 'writer')(annotator)
    return Flow([writer], flow_type = BATCH)

if __name__ == "__main__":
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()
