from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from videoflow.core.node import OneTaskProcessorNode

from .detection import Detection
from .nn_matching import NearestNeighborDistanceMetric
from .tracker import Tracker

class DeepSort(OneTaskProcessorNode):
    '''
    Deep sort algorithm implementation taken from:
    https://github.com/nwojke/deep_sort

    - Arguments:
        - min_height: Only detections of height >= min_height will be \
            considered.
        - max_cosine_distance: Gating threshold for cosine distance of \
            features
        - nn_budget: Maximum size of the appeareance descriptors gallery. \
            If None, no budget is enforced.
    '''
    def __init__(self, min_height = 0, max_cosine_distance = 0.2,
                nn_budget = None):
        self._min_height = min_height
        self._max_cosine_distance = max_cosine_distance
        self._nn_budget = nn_budget
        
        metric = NearestNeighborDistanceMetric(
            "cosine", self._max_cosine_distance, self._nn_budget
        )
        self._tracker = Tracker(metric)
        super(DeepSort, self).__init__()
    
    def process(self, bboxes):
        '''
        - Arguments:
            - bboxes (np.array) (nb_boxes, 133). \
                The 133 is splitted as follows: [top, left, width, height, confidence, features...]
        
        - Returns:
            - tracks: (np.array) (nb_tracks, 5) \
                Specifically (nb_boxes, [top, left, width, height, track_id])
        '''
        detection_list = []
        for bbox_data in bboxes:
            bbox, confidence, feature = bbox_data[0:4], bbox_data[4], bbox_data[5:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        
        results = []
        self._tracker.predict()
        self._tracker.update(detection_list)
        for track in self._tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append(
                np.array([bbox[0], bbox[1], bbox[2], bbox[3], track.track_id], np.int32)
            )
        
        return np.array(results, np.int32)
