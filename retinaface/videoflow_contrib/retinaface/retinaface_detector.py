from __future__ import print_functino
from __future__ import division
from __future__ import absolute_import

import numpy as np

from videoflow.core.node import ProcessorNode
from videoflow.core.constants import CPU, GPU
from videoflow.utils.downloader import get_file


BASE_URL_RETINAFACE = 'https://github.com/videoflow/videoflow-contrib/releases/download/retinaface/'

class RetinafaceDetector(ProcessorNode):
    '''

    '''
    def __init__(self, nb_tasks = 1, device_type = GPU):
        self._scales = [1024, 1980]
        self._thresh = 0.8
        self._detector = None
        super(Retinaface, self).__init__(nb_tasks = nb_tasks, device_type = device_type)
    
    def open(self):
        remote_url = BASE_URL_RETINAFACE + 'R50-0000.params'
        path_to_model_file = get_file('R50-0000.params', remote_url)
        if self.device_type == CPU:
            ctx_id = -1
        elif self.device_type == GPU:
            ctx_id = 0
        self._detector = RetinaFace(path_to_model_file, 0, ctx_id, 'net3')
    
    def process(self, img: np.array):
        '''
        - Parameters:
            - im: np.array of shape (h, w, 3) in BGR order on the last dimension (opencv order)

        - Returns:
            - faces: np.array of shape (nb_people, [xmin, ymin, xmax, ymax, score])
            - landmarks: np.array of shape (nb_people, 5), representing the locations \
                of eyes, nose, and left and right side of mouth.
        '''
        im_shape = img.shape
        target_size = self._scales[0]
        max_size = self._scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]

        faces, landmarks = self._detector.detect(img, self._thresh, scales = scales, do_flip = False)
        return faces.astype(int), landmarks

    def close(self):
        pass