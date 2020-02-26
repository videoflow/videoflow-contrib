from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms

from .utils import (
    bbox_overlaps,
    warp_pos,
    get_center,
    get_height,
    get_width,
    make_pos
)

class Tracker:
    '''
    The main tracking file. Here is where the magic happens
    ''' 
    def __init__(self, 
                obj_detect, 
                reid_network, 
                detection_person_thresh,
                regression_person_thresh,
                detection_nms_thresh,
                regression_nms_thresh,
                public_detections,
                inactive_patience,
                do_reid,
                max_features_num,
                reid_sim_threshold,
                reid_iou_threshold,
                do_align,
                motion_model_cfg,
                warp_mode,
                number_of_iterations,
                termination_eps
        )

        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = detection_person_thresh
        self.regression_person_thresh = regression_person_thresh
        self.detection_nms_thresh = detection_nms_thresh
        self.regression_nms_thresh = regression_nms_thresh
        self.public_detections = public_detections
        self.inactive_patience = inactive_patience
        self.do_reid = do_reid
        self.max_features_num = max_features_num
        self.reid_sim_threshold = reid_sim_threshold
        self.reid_iou_threshold = reid_iou_threshold
        self.do_align = do_align
        self.motion_model_cfg = motion_model_cfg
        self.warp_mode = warp_mode
        self.number_of_iterations = number_of_iterations
        self.termination_eps = termination_eps

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
    
    def reset(self, hard = True:
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0
    
    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
            self.inactive_tracks += tracks
    
    def add(self, new_det_pos, new_det_scores, new_det_features):
        '''
        Initializes new Track objects and saves them
        '''
        num_new = dew_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    self.track_num + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
                )
            )
        self.track_num += num_new
    
    def regress_tracks(self, blob):
        '''
        Regresses the position of the tracks and also checks their scores
        '''
        pos = self.get_pos()
        boxes, scores = self.obj_detect.predict_boxes(blob['img'], pos)
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] < self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                t.pos = pos[i].view(1, -1)
        
        return torch.Tensor(s[::-1]).cuda()
    
    def get_pos(self):
        '''
        Gets the positions of all the active tracks
        '''
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos
    
    def get_features(self):
        '''
        Get the features of all active tracks
        '''
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        
        return features
    
    def get_inactive_features(self):
        '''
        Get the features of all inactive tracks
        '''
        pass
    

class Track(object):
    '''
    This class contains all necessary for every individual track.
    '''
    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen = mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None
    
    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]
    
    def add_features(self, features):
        '''
        Adds new appearance features to the object
        '''
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()
    
    def test_features(self, test_features):
        '''
        Compares test_features to features of this Track object
        '''
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim = 0)
        else:
            features = self.features[0]
        
        features = features.mean(0, keepdim = True)
        dist = F.pairwise_distance(features, test_features, keepdim = True)
        return dist
    
    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())



