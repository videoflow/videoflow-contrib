import sys

import numpy as np

import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.annotators import TrackerAnnotator
from videoflow_contrib.tracktor import TracktorFromFrames

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

class TracksToAnnotator(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(TracksToAnnotator, self).__init__()
    
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

def main1():
    output_file = sys.argv[1]
    input_file = get_file(
        VIDEO_NAME, 
        URL_VIDEO)
    reader = VideofileReader(input_file)
    frame = FrameIndexSplitter()(reader)
    tracks = TracktorFromFrames()(frame)
    tracks_to_annotator = TracksToAnnotator()(tracks)
    annotator = TrackerAnnotator()(frame, tracks_to_annotator)
    writer = VideofileWriter(output_file, fps = 30)(annotator)
    fl = flow.Flow([reader], [writer], flow_type = BATCH)
    fl.run()
    fl.join()

def main():
    output_file = sys.argv[1]
    input_file = get_file(
        VIDEO_NAME, 
        URL_VIDEO)
    
    reader = VideofileReader(input_file)
    reader.open()
    frame_splitter = FrameIndexSplitter()
    frame_splitter.open()
    track_from_frames = TracktorFromFrames()
    track_from_frames.open()
    tracks_to_annotator = TracksToAnnotator()
    tracks_to_annotator.open()
    annotator = TrackerAnnotator()
    annotator.open()
    writer = VideofileWriter(output_file, fps = 8)
    writer.open()

    counter = 0
    while True:
        print(counter)
        counter += 1
        
        try:
            index, next_frame = reader.next()
        except:
            break
        tracks = track_from_frames.process(next_frame)
        try:
            transformed_tracks = tracks_to_annotator.process(tracks)
        except Exception as e:
            print(tracks.shape)
            continue
        annotated_frame = annotator.process(next_frame, tracks)
        writer.consume(annotated_frame)

    reader.close()
    frame_splitter.close()
    track_from_frames.close()
    tracks_to_annotator.close()
    annotator.close()
    writer.close()
    
    
    
if __name__ == "__main__":
    main1()