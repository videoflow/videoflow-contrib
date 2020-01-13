import numpy as np
import mxnet as mx

from videoflow.core import ProcessorNode
from videoflow.core.constants import GPU


class MxnetPersonReid(ProcessorNode):
    """
    Returns persons ids based on features matrix for several persons.

    Ids are given sequentially, starting from 0. Uses MXNet for distance calc.

    - Arguments:
        - min_similarity_threshold (float): person re-id  will give a new id \
            for entries with similarity below threshold similarity.
    """

    def __init__(self, min_similarity_threshold: float = 0.5, nb_tasks=1, device_type=GPU):
        self._context = mx.gpu() if device_type == GPU else mx.cpu()
        self._similarity_threshold = min_similarity_threshold
        self._features = None
        super().__init__(nb_tasks=nb_tasks, device_type=device_type)

    def process(self, features: np.array) -> np.array:
        """Returns persons ids based on features matrix for several persons.

        Ids are given sequentially, starting from 0.

        - Arguments:
            - features: np.array of shape (batch, feature_vector_length)

        - Returns:
            - ids: np.array of shape (batch, )
        """
        if features.shape[0] == 0:
            return np.empty((0, ), int)
        features = mx.nd.array(features).as_in_context(self._context)
        if self._features is None:
            self._features = features.copy()
            return np.arange(0, features.shape[0])
        dist_all = mx.nd.linalg.gemm2(features, self._features, transpose_b=True)
        result = np.empty((features.shape[0], ), int)
        for i in range(features.shape[0]):
            person_id: int = dist_all[i].argsort(is_ascend=False).as_in_context(mx.cpu()).asnumpy().astype("int32")[0]
            if dist_all[i][person_id] < self._similarity_threshold:
                self._features = mx.nd.concat(self._features, features, dim=0)
                result[i] = self._features.shape[0] - 1
            else:
                self._features[person_id, :] = features[i, :].copy()
                result[i] = person_id
        return result
