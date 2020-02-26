from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from videoflow.core.node import OneTaskProcessorNode


class Tracktor(OneTaskProcessorNode):
    '''
    Tracktor algorithm with REID taken from
    https://github.com/phil-bergmann/tracking_wo_bnw
    '''

    def __init__(self):
        super(Tracktor, self).__init__()
    
    def open(self):
        '''
        Load models here.
        '''
        pass
    
    def process(self, bboxes):
        '''
        - Arguments:
            - bboxes (np.array) (nb_boxes, 5) The 5 should be interpreted as [top, left, width, height, score]
        
        - Returns:
            - tracks: (np.array) (nb_boxes, 5) \
                Specifically (nb_boxes, [top, left, width, height, track_id])
        '''
        return bboxes
