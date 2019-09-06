from typing import List

import cv2
import mxnet as mx
import numpy as np

from videoflow.core import ProcessorNode
from videoflow.core.constants import GPU
from .gluoncv_person_reid import resnet50, get_transform


class MxnetPersonFeatureExtractor(ProcessorNode):
    """
    Calculates a feature vector for a person on the image that can be used to
    compute similarity metrics on it.

    See https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline on
    how to build params file.
    """

    FEATURE_VECTOR_LEN = 2048

    def __init__(self, path_to_params_file: str, nb_tasks=1, device_type=GPU):
        self._mxnet_model = None
        self._transform = None
        self._context = mx.gpu() if device_type == GPU else mx.cpu()
        self._path_to_params_file = path_to_params_file
        super().__init__(nb_tasks=nb_tasks, device_type=device_type)

    def open(self):
        self._transform = get_transform()
        self._mxnet_model = resnet50(ctx=self._context, pretrained=False)
        self._mxnet_model.load_parameters(
            self._path_to_params_file,
            ctx=self._context,
            allow_missing=True,
            ignore_extra=True,
        )

    @staticmethod
    def _flip_horizontally(im_batch: mx.nd.NDArray) -> mx.nd.NDArray:
        """Flip image batch horizontally.

        - Arguments:
            - im_batch: mxnet.nd.NDArray of shape (batch, h, w, 3)

        - Returns:
            - flipped_im_batch: mxnet.nd.NDArray of shape (batch, h, w, 3)
        """
        return mx.nd.flip(im_batch, axis=3)

    def process(self, im_batch: List[np.array]) -> np.array:
        """
        - Arguments:
            - im_batch: list (with the length of batch) of np.array of shape \
                (h, w, 3)

        - Returns:
            - features: np.array of shape (batch, feature_vector_length)
        """
        batch_size = len(im_batch)
        result = np.zeros((batch_size, self.FEATURE_VECTOR_LEN))
        if batch_size == 0:
            return result

        im_nd_list = []
        for im in im_batch:
            im = mx.nd.array(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype('uint8')
            im = self._transform(im)
            im = mx.nd.expand_dims(im, axis=0)
            im_nd_list.append(im)
        im_batch = mx.nd.concat(*im_nd_list, dim=0)
        for j in range(2):
            if j == 1:
                im_batch = self._flip_horizontally(im_batch)
            f = self._mxnet_model(im_batch.as_in_context(self._context)).as_in_context(mx.cpu()).asnumpy()
            result += f
        return result / np.linalg.norm(result, axis=1, keepdims=True)
