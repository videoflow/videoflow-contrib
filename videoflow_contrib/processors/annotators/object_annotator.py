from typing import Optional, Mapping

import cv2
import numpy as np

from videoflow.core import ProcessorNode


class ObjectAnnotator(ProcessorNode):
    """
    Draws bounding boxes around objects on images.

    - Arguments:
        - class_labels: mapping from int id to str label of class
        - box_color: color to use to draw the boxes
        - box_thickness: thickness of boxes to draw
        - text_color: color of text to draw
    """

    def __init__(self,
                 class_labels: Mapping[int, str],
                 box_color=(255, 225, 0),
                 box_thickness=2,
                 text_color=(255, 255, 0),
                 nb_tasks=1,
                 ):
        self._class_labels = class_labels
        self._box_color = box_color
        self._text_color = text_color
        self._box_thickness = box_thickness
        super().__init__(nb_tasks=nb_tasks)

    def _annotate(self,
                  im: np.array,
                  boxes: np.array,
                  class_ids: Optional[np.array] = None,
                  confidence: Optional[np.array] = None,
                  ids: Optional[np.array] = None,
                  ) -> np.array:
        """Annotates image. See `process` for arguments and return value docs."""

        for i in range(len(boxes)):
            bbox = boxes[i]
            ymin, xmin, ymax, xmax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            y_label = ymin - 15 if ymin - 15 > 15 else min(ymin + 15, ymax)
            class_label = self._class_labels.get(class_ids[i])
            if class_label is None:
                class_label = ''
            if confidence is not None:
                confidence_i = confidence[i]
                confidence_str = f': {confidence_i * 100:.2f}%'
            else:
                confidence_str = ''
            if ids is not None:
                id_ = ids[i]
                id_str = f' {id_}'
            else:
                id_str = ''
            label = f"{class_label}{id_str}{confidence_str}"
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), self._box_color, self._box_thickness)
            cv2.putText(im, label, (xmin, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._text_color,
                        lineType=cv2.LINE_AA)
        return im

    def process(self,
                im: np.array,
                boxes: np.array,
                class_ids: Optional[np.array] = None,
                confidence: Optional[np.array] = None,
                ids: Optional[np.array] = None,
                ) -> np.array:
        """
        - Arguments:
            - im: np.array
            - boxes: np.array of shape (batch, 4) \
                second dimension entries are [ymin, xmin, ymax, xmax]
            - class_ids: np.array of shape (batch, ) or None
            - confidence: np.array of shape (batch, ) or None
            - ids: np.array of shape (batch, ) or None

        - Returns:
            - annotated_im: image with the visual annotations embedded in it.
        """

        to_annotate = im.copy()
        return self._annotate(
            im=to_annotate,
            boxes=boxes,
            class_ids=class_ids,
            confidence=confidence,
            ids=ids,
        )
