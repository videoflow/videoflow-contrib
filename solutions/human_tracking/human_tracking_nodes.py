'''
Glue nodes for the human-tracking solution.

These live in their own importable module (not in the graph module) because
distributed workers reconstruct each node from its class path — recorded as
``human_tracking_nodes.<Class>`` — and importing the graph module in a worker
would re-run graph-level code. Every constructor argument is stored and
forwarded through ``**kwargs`` so the node round-trips through the wire.
'''
from __future__ import annotations

import numpy as np
from videoflow.core.node import ProcessorNode


class FrameIndexSplitter(ProcessorNode):
    '''Drops the frame index from a ``(index, frame)`` producer tuple.'''

    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame


class KeypointsExtractor(ProcessorNode):
    '''Takes the keypoints half of the pose node's ``(keypoints, boxes)`` output.'''

    def __init__(self, **kwargs):
        super(KeypointsExtractor, self).__init__(**kwargs)

    def process(self, data):
        keypoints, bounding_boxes = data
        return keypoints


class BoundingBoxesExtractor(ProcessorNode):
    '''
    Takes the boxes half of the pose node's output and converts them from
    ``[ymin, xmin, ymax, xmax]`` corners to the ``[ymin, xmin, width, height]``
    form DeepSort expects, appending a unit score column.
    '''

    def __init__(self, **kwargs):
        super(BoundingBoxesExtractor, self).__init__(**kwargs)

    def process(self, data):
        '''
        - Returns:
            - bounding_boxes: (nb_boxes, [ymin, xmin, width, height, score])
        '''
        keypoints, bounding_boxes = data
        scores = np.ones((bounding_boxes.shape[0], 1))
        bounding_boxes = np.concatenate(
            [bounding_boxes[:, [1, 0]],
            np.expand_dims(bounding_boxes[:, 2] - bounding_boxes[:, 0], 1),
            np.expand_dims(bounding_boxes[:, 3] - bounding_boxes[:, 1], 1)], axis = 1)
        bounding_boxes = np.concatenate([bounding_boxes, scores], axis = 1)
        bounding_boxes = bounding_boxes.astype(np.int32)
        return bounding_boxes


class CropBoundingBoxes(ProcessorNode):
    '''Crops each person's box out of the frame, for the appearance encoder.'''

    def __init__(self, **kwargs):
        super(CropBoundingBoxes, self).__init__(**kwargs)

    def process(self, im, bounding_boxes):
        '''
        - Arguments:
            - im: np.array of shape (h, w, 3)
            - bounding_boxes: np.array of shape (nb_boxes, [ymin, xmin, width, height, score])

        - Returns:
            - im_list: list of np.array (h, w, 3)
        '''
        to_return = []
        for bbox in bounding_boxes:
            ymin, xmin, width, height, _ = bbox
            crop = im[ymin:ymin + height, xmin:xmin + width, :]
            to_return.append(crop)
        return to_return


class AppendFeaturesToBoundingBoxes(ProcessorNode):
    '''Concatenates each box with its appearance feature vector — DeepSort's input.'''

    def __init__(self, **kwargs):
        super(AppendFeaturesToBoundingBoxes, self).__init__(**kwargs)

    def process(self, bboxes, features):
        '''
        - Arguments:
            - bboxes: (batch, [ymin, xmin, width, height, score])
            - features: (batch, nb_features)
        '''
        to_return = np.concatenate([bboxes, features], axis = 1)
        return to_return


class ConvertTracksForAnotation(ProcessorNode):
    '''Converts tracks back from ``[ymin, xmin, width, height]`` to corner form for the annotator.'''

    def __init__(self, **kwargs):
        super(ConvertTracksForAnotation, self).__init__(**kwargs)

    def process(self, tracks):
        '''
        - Arguments:
            - tracks: (nb_tracks, [ymin, xmin, width, height, track_id])

        - Returns:
            - tracks: (nb_tracks, [ymin, xmin, ymax, xmax, track_id])
        '''
        if len(tracks) == 0:
            return tracks
        # Column-stack keeps each term 1-D; np.concatenate on axis 1 would need
        # every part to be 2-D and silently fails on the derived columns.
        to_return = np.column_stack([
            tracks[:, 0],
            tracks[:, 1],
            tracks[:, 0] + tracks[:, 3],
            tracks[:, 1] + tracks[:, 2],
            tracks[:, 4],
        ]).astype(np.int32)
        return to_return
