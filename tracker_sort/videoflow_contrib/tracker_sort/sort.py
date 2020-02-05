from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from videoflow.processors.vision.trackers import BoundingBoxTracker
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
import math

def eucl(bb_test, bb_gt):
    '''
    Computes the euclidean distance between two boxes
    in the form [x1, y1, x2, y2]
    '''
    center_1 = [(bb_test[0] + bb_test[2]) / 2.0, (bb_test[1] + bb_test[3]) / 2.0]
    center_2 = [(bb_gt[0] + bb_gt[2]) / 2.0, (bb_gt[1] + bb_gt[3]) / 2.0]
    eucl = math.sqrt((center_1[0] - center_2[0])*(center_1[0] - center_2[0]) + (center_1[1] - center_2[1])*(center_1[1] - center_2[1]))
    return -eucl

def iou(bb_test, bb_gt):
    """
      Computes IUO between two bboxes in the form [y1, x1, y2, x2]
      IOU is the intersection of areas.
    """
    yy1 = np.maximum(bb_test[0], bb_gt[0])
    xx1 = np.maximum(bb_test[1], bb_gt[1])
    yy2 = np.minimum(bb_test[2], bb_gt[2])
    xx2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return(o)

def metric_factory(metric_type):
    if metric_type == "iou":
        return iou
    elif metric_type == "euclidean":
        return eucl
    else:
        raise ValueError("Cannot identify metric_type {}".format(metric_type))

def convert_bbox_to_z(bbox):
    """
      Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
        [x, y, s, r] where x, y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    x = bbox[1] + w/2.
    y = bbox[0] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    '''
    Takes a bounding box in the form [x, y, s, r] and returns it in the form
    [y1, x1, y2, x2] where x1, y1 is the top left and x2, y2 is the bottom right
    '''
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[1] - h/2., x[0] - w/2., x[1] + h/2., x[0] + w/2.]).reshape((1, 4))
    else:
        return np.array([x[1] - h/2., x[0] - w/2., x[1] + h/2., x[0] + w/2., score]).reshape((1, 5))

def associate_detections_to_trackers(detections, trackers, metric_function, iou_threshold = 0.1):
    """
      Assigns detections to tracked object (both represented as bounding boxes)
      Returns 3 lists of matches, unmatched_detections and unmatched_trackers
      - Returns:
        - matches: np.array of shape (n, 2)
        - unmatched_detections: np.array of shape (nb_of_unmatches, ) that contains the 
            indices of the detections that were not matched
        - unmatched_trackers: np.array of shape (nb_of_nonmatched_tracks, ) that contains the
            indices of the unmatched tracks
    """
    distance_threshold = 500

    if len(trackers) == 0:
        return np.empty((0, 2), dtype = int), np.arange(len(detections)), np.empty((0, 5), dtype = int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = metric_function(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    
    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
        #if(iou_matrix[m[0], m[1]] > distance_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype = int)
    else:
        matches = np.concatenate(matches, axis = 0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class KalmanBoxTracker(object):
    """
      This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class KalmanFilterBoundingBoxTracker(BoundingBoxTracker):
    '''
    - Arguments:
        - max_age: If no bounding box is matched to an internal tracklet for ``max_age`` steps \
            the internal tracklet is considered dead and is removed.
        - min_hits: A tracklet is considered a valid track if it has a hit streak larger \
            than or equal to ``min_hits``
        - metric_function_type : str, one of ``iou`` or ``euclidean`` 
        - show_in_between: If set to True, it will return tracks of bounding boxes even if they were
            not detected by the detector.  If set to False, it will return only the tracks for the 
            bounding boxes detected in this frame by the detector. Note that when set to False, it
            will return the same number of entries and in the same order as they were passed to it.
        - return_original_dets: Returns the same number of tracks as original dets, with indexes matching
            If for some reason an original det does not have a corresponding track, the track index is -1
    '''
    
    def __init__(self, max_age = 7, min_hits = 3, metric_function_type = 'iou', show_in_between = True, return_original_dets = False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.metric_function_type = metric_function_type
        self.previous_fid = -1
        self.return_original_dets = return_original_dets
        self.show_in_betwee = show_in_between
        self.metric_function = metric_factory(metric_function_type)
        super(KalmanFilterBoundingBoxTracker, self).__init__()

    def _track(self, dets, fid = None):
        """
        Requires: this method must be called once for each frame even with empty detections.

        - Arguments:
            - dets: a numpy array of detections in the format [[ymin,xmin,ymax,xmax,score],[ymin,xmin,ymax,xmax,score],...]
                
        - Returns:
            - A similar array, where the last column is the object or track id.  The number of objects returned may differ from the number of detections provided.
        """
        if fid is None:
            fid = self.previous_fid + 1
        
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.metric_function)

        #update matched trackers with assigned detections
        d_to_t = {}
        for t, trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1] == t)[0], 0]
                if len(d) > 0:
                    d_to_t[d[0]] = t
                trk.update(dets[d,:][0])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)

        ret = []
        if self.return_original_dets:
            for idx, det in enumerate(dets):
                if idx in d_to_t:
                    trk = self.trackers[d_to_t[idx]]
                    d = trk.get_state()[0]
                    ret.append(np.concatenate((d,[trk.id + 1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
                else:
                    ret.append(np.concatenate((det[:4], [-1])).reshape(1, -1))
        else:
            for trk in self.trackers:
                d = trk.get_state()[0]
                if self.show_in_between:
                    if((trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                        ret.append(np.concatenate((d,[trk.id + 1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
                else:
                    if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                        ret.append(np.concatenate((d,[trk.id + 1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
        
        # Remove dead tracklets
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if(len(ret) > 0):
            return np.concatenate(ret)
        
        self.previous_fid = fid
        return np.empty((0, 5))
