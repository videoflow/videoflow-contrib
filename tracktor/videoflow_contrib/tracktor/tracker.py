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
                motion_model_cfg,
                warp_mode,
                number_of_iterations,
                termination_eps,
                do_align = False
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
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features
    
    def reid(self, blob, new_det_pos, new_det_scores):
        pass
    
    def get_appearances(self, blob):
        pass

    def add_features(self, new_features):
        pass
    
    def align(self, blob):
        pass

    def motion_step(self, track):
        pass
    
    def motion(self):
        pass
    
    def step(self, blob):
        '''
        This function should be called at every timestep to perform tracking
        with a blob containing the image information.
        '''
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())
        
        # 1. Look for new detections
        if self.public_detections:
            dets = blob['dets'].squeeze(dim = 0)
            if dets.nelement() > 0:
                # TODO: I don't understand why the predict_boxes method needs to be 
                # called here if the boxes are already provided.
                boxes, scores = self.obj_detect.predict_boxes(blob['img'], dets)
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])
        
        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()
        
        if inds.nelement() > 0:
            det_pos = boxes[inds]
            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()
        
        # 2. Predict tracks
        num_tracks = 0
        nms_inp_reg = torch.zeros(0).cuda()
        if len(self.tracks):
            # 2.1 Align
            if self.do_align:
                self.align(blob)

            # 2.2 Apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]
            
            # 2.3 Regress
            person_scores = self.regress_tracks(blob)
            if len(self.tracks):
                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)
                self.tracks_to_inactive([self.tracks[i] for in in list(range(len(self.tracks))) if i not in keep])
                if keep.nelement() > 0:
                    if self.do_reid:
                        new_features = self.get_appearances(blob)
                        self.add_features(new_features)
        
        # 3. Create new tracks
        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_threshold)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # Check with every track in a single run (problem if tracks delete each other)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores]
                )
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)
                keep = keep[torch.ge(keep, 1)] - 1
                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break
        
        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # Try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)
            
        
        # 4. Generate results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])
        
        for t in self.inactive_tracks:
            t.count_inactive += 1
        
        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]
    
    def get_results(self):
        return self.results
    

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



