from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from videoflow.core.node import ProcessorNode
from videoflow.core.constants import CPU, GPU
from videoflow.processors.vision.segmentation import Segmenter
from ...utils.tensorflow import TensorflowModel
from videoflow.utils.downloader import get_file

import tensorflow as tf

BASE_URL_SEGMENTATION = 'https://github.com/videoflow/videoflow-contrib/releases/download/segmentation_tf/'

class TensorflowSegmenter(Segmenter):
    '''
    Finds masks by running a Tensorflow model on an image.

    Initializes the tensorflow model.  If ``path_to_pb_file`` is provided, it uses a local
    model. If not, it uses ``architecture`` and ``dataset`` parameters to download tensorflow
    pretrained models.  

    .. csv-table:: Models supported COCO dataset

        "Model","Speed (ms)","COCO mAP"
        "maskrcnn-resnet101_coco","470","33"
        "maskrcnn-inceptionv2_coco","79","25"


    - Arguments:
        - num_classes (int): The number of classes that the segmenter can recognize.
        - path_to_pb_file (str): Path where model pb file is \
            It expects the model to have the following input tensors: ``image_tensor:0``, and \
            the following output tensors: ``detection_boxes:0``, ``detection_scores:0``, \
            ``detection_classes:0``, ``num_detections:0`` and ``detection_masks:0``.  If no path is provided, then \
            it will download the model from the internet using the values provided for ``architecture``\
            and ``dataset``.
        - architecture (str): One of the architectures mentioned in the tables above is accepted.
        - dataset (str): For now, only `coco` is accepted.
        - min_score_threshold (float): detection will filter out entries with score below threshold score
    '''
    supported_models = [
        "maskrcnn-resnet101_coco",
        "maskrcnn-inceptionv2_coco"
    ]

    def __init__(self, 
                num_classes = 90,
                path_to_pb_file = None,
                architecture = 'maskrcnn-inceptionv2',
                dataset = 'coco',
                min_score_threshold = 0.5,
                nb_tasks = 1,
                device_type = GPU):
        self._num_classes = num_classes
        self._path_to_pb_file = path_to_pb_file
        
        if path_to_pb_file is None and (architecture is None or dataset is None):
            raise ValueError('If path_to_pb_file is None, then architecture and dataset cannot be None')

        if path_to_pb_file is None:
            remote_model_id = f'{architecture}_{dataset}'
            if remote_model_id not in self.supported_models:
                raise ValueError('model is not one of supported models: {}'.format(', '.join(self.supported_models)))        
            self._remote_model_file_name = f'{architecture}_{dataset}.pb'

        self._min_score_threshold = min_score_threshold
        super(TensorflowSegmenter, self).__init__(nb_tasks = nb_tasks, device_type = device_type)
    
    def open(self):
        '''
        Creates session with tensorflow model
        '''
        if self.device_type == CPU:
            device_id = 'cpu'
        elif self.device_type == GPU:
            device_id = 'gpu'
        else:
            device_id = 'cpu'
        
        if self._path_to_pb_file is None:
            remote_url = BASE_URL_SEGMENTATION + self._remote_model_file_name
            self._path_to_pb_file = get_file(self._remote_model_file_name, remote_url)
        
        with tf.device(device_id):
            self._model_graph = tf.Graph()
            with self._model_graph.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.GFile(self._path_to_pb_file, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name = '')
        self._session = tf.Session(graph = self._model_graph)
        self._detection_boxes = self._model_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_masks = self._model_graph.get_tensor_by_name('detection_masks:0')
        self._num_detections = self._model_graph.get_tensor_by_name('num_detections:0')
        self._detection_scores = self._model_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = self._model_graph.get_tensor_by_name('detection_classes:0')
        self._image_tensor = self._model_graph.get_tensor_by_name('image_tensor:0')
    
    def close(self):
        '''
        Closes tensorflow model session.
        '''
        if self._session:
            self._session.close()

    def _segment(self, im : np.array) -> np.array:
        '''
        - Arguments:
            - im (np.array): (h, w, 3)
        
        - Returns:
            - masks: np.array of shape (nb_masks, h, w)
            - classes: np.array of shape (nb_masks,)
            - scores: np.array of shape (nb_masks,)
        '''
        h, w, _ = im.shape
        im_expanded = np.expand_dims(im, axis = 0)
        detection_boxes = tf.squeeze(self._detection_boxes, [0])
        detection_masks = tf.squeeze(self._detection_masks, [0])
        real_num_detection = tf.cast(self._num_detections[0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            detection_masks,
            detection_boxes,
            h,
            w
        )
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        detection_masks_reframed = tf.expand_dims(detection_masks_reframed, 0)
        masks, scores, classes = self._session.run(
            [detection_masks_reframed, self._detection_scores, self._detection_classes],
            feed_dict = {
                self._image_tensor: im_expanded
            }
        )
        masks, scores, classes = np.squeeze(masks, axis = 0), np.squeeze(scores, axis = 0), np.squeeze(classes, axis = 0)
        indexes = np.where(scores > self._min_score_threshold)[0]
        masks, scores, classes = masks[indexes], scores[indexes], classes[indexes]
        return masks, classes, scores
