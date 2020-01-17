import numpy as np
import cv2
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from videoflow.core.node import ProcessorNode
from videoflow.core.constants import GPU, CPU
from videoflow.utils.downloader import get_file

BASE_URL_DETECTRON2 = 'https://github.com/videoflow/videoflow-contrib/releases/download/detectron2/'

class HumanPoseAnnotator(ProcessorNode):
    _KEYPOINT_THRESHOLD = 0.05
    
    COCO_PERSON_KEYPOINT_NAMES = (
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    )

    COCO_PERSON_KEYPOINT_FLIP_MAP = (
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle")
    )

    KEYPOINT_CONNECTION_RULES = [
        # face
        ("left_ear", "left_eye", (102, 204, 255)),
        ("right_ear", "right_eye", (51, 153, 255)),
        ("left_eye", "nose", (102, 0, 204)),
        ("nose", "right_eye", (51, 102, 255)),
        # upper-body
        ("left_shoulder", "right_shoulder", (255, 128, 0)),
        ("left_shoulder", "left_elbow", (153, 255, 204)),
        ("right_shoulder", "right_elbow", (128, 229, 255)),
        ("left_elbow", "left_wrist", (153, 255, 153)),
        ("right_elbow", "right_wrist", (102, 255, 224)),
        # lower-body
        ("left_hip", "right_hip", (255, 102, 0)),
        ("left_hip", "left_knee", (255, 255, 77)),
        ("right_hip", "right_knee", (153, 255, 204)),
        ("left_knee", "left_ankle", (191, 255, 128)),
        ("right_knee", "right_ankle", (255, 195, 77))
    ]

    def __init__():
        super(HumanPoseAnnotator, self).__init__()

    def process(self, im: np.array, keypoints: np.array):
        '''
        - Arguments: 
            - im: np.array of shape (h, w, 3)
            - keypoints: np.array of shape (nb_people, 17, 3)
        
        - Returns:
            - im: np.array of shape (h, w, 3) annotated with keypoints on it.
        '''
        visible = {}
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > self._KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=_RED)
                if keypoint_names:
                    keypoint_name = self.COCO_PERSON_KEYPOINT_NAMES[idx]
                    visible[keypoint_name] = (x, y)
        
        # Draw normal connected lines
        for kp0, kp1, color in self.KEYPOINT_CONNECTION_RULES:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                im = cv2.line(im, (x0, y0), (x1, y1), color, thickness = 10)
        
        # Draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        try:
            ls_x, ls_y = visible["left_shoulder"]
            rs_x, rs_y = visible["right_shoulder"]
            mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        except KeyError:
            pass
        else:
            nose_x, nose_y = visible.get("nose", (None, None))
            if nose_x is not None:
                im = cv2.line(im, (nose_x, nose_y), (mid_shoulder_x, mid_shoulder_y), color = (0, 0, 255))
           
            try:
                # draw line from mid-shoulder to mid-hip
                lh_x, lh_y = visible["left_hip"]
                rh_x, rh_y = visible["right_hip"]
            except KeyError:
                pass
            else:
                mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                im = cv2.line(im, (mid_hip_x, mid_hip_y), (mid_shoulder_x, mid_shoulder_y), color = (0, 0, 255))
        
        return im

class Detectron2HumanPose(ProcessorNode):
    def __init__(self, path_to_model_file = None, path_to_model_config = None, architecture = None,
                nb_tasks = 1, device_type = GPU):
        self._path_to_model_file = path_to_model_file
        self._path_to_model_config = path_to_model_config
        self._predictor = None
        if self._path_to_model_file is None and architecture is None:
            raise ValueError('If path_to_model_file is None, then architecture cannot be None')
        if self._path_to_model_file is not None and path_to_model_config is None:
            raise ValueError('If path_to_model_file is not None, then path_to_model_config is not None')
        if self._path_to_model_file is None:
            self._remote_model_file_name = f'{architecture}.pkl'
            self._remote_model_config_name = f'{architecture}.yaml'
        super(Detectron2HumanPose, self).__init__(nb_tasks = nb_tasks, device_type = device_type)

    def open(self):
        if self._path_to_model_file is None:
            remote_model_url = BASE_URL_DETECTRON2 + self._remote_model_file_name
            remote_config_url = BASE_URL_DETECTRON2 + self._remote_model_config_name
            self._path_to_model_file = get_file(self._remote_model_file_name, remote_model_url)
            self._path_to_model_config = get_file(self._remote_model_config_name, remote_config_url)
        cfg = get_cfg()
        cfg.merge_from_file(self._path_to_model_config)
        cfg.MODEL.WEIGHTS = self._path_to_model_file
        if self.device_type == CPU:
            cfg.MODEL.DEVICE = 'cpu'
        elif self.device_type == GPU:
            cfg.MODEL.DEVICE = 'gpu'
        self._predictor = DefaultPredictor(cfg)
    
    def process(self, im: np.array):
        '''
        - Parameters:
            - im: np.array of shape (h, w, 3) in BGR order on the last dimension (opencv order)
        
        - Returns:
            - pred_keypoints: np.array of shape (nb_people, 17, [x, y, prob])
            - pred_boxes: np.array of shape (nb_people, [xmin, ymin, xmax, ymax])
        '''
        outputs = predictor(im)
        in_cpu = outputs['instances'].to('cpu')
        pred_keypoints = in_cpu.pred_keypoints.numpy()
        pred_boxes = in_cpu.pred_boxes.tensor.numpy()
        return pred_keypoints, pred_boxes

    def close(self):
        pass