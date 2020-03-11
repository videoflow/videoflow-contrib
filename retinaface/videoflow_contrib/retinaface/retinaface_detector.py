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
        super(Retinaface, self).__init__(nb_tasks = nb_tasks, device_type = device_type)
    
    def open(self):
        pass
    
    def process(self, im: np.array):
        '''
        - Parameters:
            - im: np.array of shape (h, w, 3) in BGR order on the last dimension (opencv order)

        - Returns:
            - pred_boxes: np.array of shape (nb_people, [xmin, ymin, xmax, ymax, score])
        '''
        pass

    def close(self):
        pass