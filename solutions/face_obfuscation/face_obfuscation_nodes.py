'''
Glue nodes for the face-obfuscation solution.

These live in their own importable module (not in the graph module) because
distributed workers reconstruct each node from its class path — recorded as
``face_obfuscation_nodes.<Class>`` — and importing the graph module in a worker
would re-run graph-level code. Every constructor argument is stored and
forwarded through ``**kwargs`` so the node round-trips through the wire.
'''
from __future__ import annotations

import cv2
import numpy as np
from videoflow.core.node import ProcessorNode


class FrameIndexSplitter(ProcessorNode):
    '''Drops the frame index from a ``(index, frame)`` producer tuple.'''

    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame


class BoundingboxObfuscator(ProcessorNode):
    '''
    Gaussian-blurs every detected and tracked face box.

    - Arguments:
        - expand: fraction each box is grown by on all sides before blurring, so \
            the blur covers hair/chin past the detector's tight box.
        - kernel: Gaussian kernel size in pixels (forced odd, as OpenCV requires).
        - sigma: Gaussian standard deviation; higher is blurrier.
    '''

    def __init__(self, expand=0.20, kernel=23, sigma=30, nb_tasks=1, **kwargs):
        self._expand = float(expand)
        self._kernel = int(kernel) | 1          # cv2 requires an odd kernel size
        self._sigma = float(sigma)
        super(BoundingboxObfuscator, self).__init__(nb_tasks=nb_tasks, **kwargs)

    def process(self, im: np.ndarray, bounding_boxes: np.ndarray, tracker_bounding_boxes: np.ndarray):
        '''
        - Arguments:
            - im: np.array of shape (h, w, 3)
            - bounding_boxes: np.array of shape (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
            - tracker_bounding_boxes: np.array of shape (nb_boxes, [[ymin, xmin, ymax, xmax, tracker_id]])
        '''
        result_image = im.copy()

        # Blur the union of raw detections and tracker boxes: the tracker fills in
        # faces the detector missed in this frame.
        boxes = np.concatenate((bounding_boxes[:, 0:4], tracker_bounding_boxes[:, 0:4]))
        heights = boxes[:, 2] - boxes[:, 0]
        widths = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] -= self._expand * heights
        boxes[:, 2] += self._expand * heights
        boxes[:, 1] -= self._expand * widths
        boxes[:, 3] += self._expand * widths
        boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], im.shape[0])
        boxes[:, 3] = np.minimum(boxes[:, 3], im.shape[1])
        boxes = boxes.astype(np.int32)

        for box in boxes:
            ymin, xmin, ymax, xmax = box
            sub_face = im[ymin:ymax, xmin:xmax, :]
            if sub_face.size == 0:
                continue                        # box collapsed to zero area at the frame edge
            sub_face = cv2.GaussianBlur(sub_face, (self._kernel, self._kernel), self._sigma)
            result_image[ymin:ymax, xmin:xmax, :] = sub_face
        return result_image
