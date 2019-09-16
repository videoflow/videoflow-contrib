import cv2
import gluoncv as gcv
import mxnet as mx
import numpy as np

from videoflow.core.constants import GPU
from videoflow.processors.vision.detectors import ObjectDetector


class MxnetObjectDetector(ObjectDetector):
    """Finds object detections by running a MXNet model on an image."""

    AVAILABLE_ARCHITECTURES = [
        'ssd_512_vgg16_atrous_coco',
        'ssd_512_resnet50_v1_coco',
        'ssd_512_mobilenet1.0_coco',
    ]

    DEFAULT_ARCHITECTURE = 'ssd_512_mobilenet1.0_coco'

    def __init__(self,
                 architecture: str = DEFAULT_ARCHITECTURE,
                 min_score_threshold: float = 0.5,
                 nb_tasks: int = 1,
                 device_type: str = GPU,
                 ):
        if architecture not in self.AVAILABLE_ARCHITECTURES:
            raise NotImplementedError(f"{architecture} model is not supported!")

        self._mxnet_model_name = architecture
        self._min_score_threshold = min_score_threshold
        self._ctx = mx.gpu() if device_type == GPU else mx.cpu()
        self._mxnet_model = None
        super().__init__(nb_tasks=nb_tasks, device_type=device_type)

    def open(self):
        self._mxnet_model = gcv.model_zoo.get_model(self._mxnet_model_name, pretrained=True, ctx=self._ctx)

    def _detect(self, im: np.array) -> np.array:
        im_t = mx.nd.array(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, im_t = gcv.data.transforms.presets.ssd.transform_test(im_t, short=512)

        class_indexes, scores, bounding_boxes = self._mxnet_model(rgb_nd.as_in_context(self._ctx))

        class_indexes = class_indexes.asnumpy()[0]
        class_indexes = class_indexes[class_indexes > -1]
        detection_count = class_indexes.shape[0]
        class_indexes = class_indexes
        scores = scores.asnumpy()[0, :detection_count, 0]
        confidence_detection_mask = scores > self._min_score_threshold
        class_indexes = class_indexes[confidence_detection_mask]
        scores = scores[confidence_detection_mask]
        bounding_boxes = bounding_boxes.asnumpy()[0][:detection_count][confidence_detection_mask]
        width, height = im.shape[1], im.shape[0]
        width_t, height_t = im_t.shape[1], im_t.shape[0]
        bounding_boxes = gcv.data.transforms.bbox.resize(bounding_boxes, (width_t, height_t), (width, height))
        bounding_boxes[bounding_boxes < 0] = 0
        bounding_boxes[:, 0][bounding_boxes[:, 0] > width] = width
        bounding_boxes[:, 2][bounding_boxes[:, 2] > width] = width
        bounding_boxes[:, 1][bounding_boxes[:, 1] > height] = height
        bounding_boxes[:, 3][bounding_boxes[:, 3] > height] = height
        result = np.empty([sum(confidence_detection_mask), 6], float)
        result[:, 0] = bounding_boxes[:, 1]
        result[:, 1] = bounding_boxes[:, 0]
        result[:, 2] = bounding_boxes[:, 3]
        result[:, 3] = bounding_boxes[:, 2]
        result[:, 4] = class_indexes[:]
        result[:, 5] = scores[:]

        return result
