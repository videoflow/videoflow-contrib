from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import cv2
import tensorflow as tf

from videoflow.core.node import ProcessorNode
from videoflow.utils.tensorflow import TensorflowModel
from videoflow.utils.downloader import get_file
from videoflow.core.constants import CPU, GPU

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

class HumanEncoder(ProcessorNode):
    '''
    Encodes a human in a bounding box into a feature vector that 
    can be used to compute similarity metrics on it.

    Model from paper ``Deep Cosine Metric Learning for Person Re-identification`` trained
    on the ``MARS`` dataset.
    
    https://github.com/nwojke/cosine_metric_learning
    '''
    def __init__(self, path_to_pb_file = None, batch_size = 32, nb_tasks = 1, device_type = GPU):
        self._tensorflow_model = None
        self._path_to_pb_file = path_to_pb_file
        self._batch_size = batch_size
        super(HumanEncoder, self).__init__(nb_tasks = nb_tasks, device_type = device_type)
    
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
            remote_url = 'https://github.com/videoflow/videoflow-contrib/releases/download/models/humanencoder_mars_128.pb'
            self._path_to_pb_file = get_file('human_encoder.pb', remote_url)
        
        with tf.device(device_id):
            self._model_graph = tf.Graph()
            with self._model_graph.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.GFile(self._path_to_pb_file, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name = '')
                
        self._session = tf.Session(graph = self._model_graph)
        self._input_var = self._model_graph.get_tensor_by_name('images:0')
        self._output_var = self._model_graph.get_tensor_by_name('features:0')
        self._feature_dim = self._output_var.get_shape().as_list()[-1]
        self._image_shape = self._input_var.get_shape().as_list()[1:]
        
    def close(self):
        '''
        Closes tensorflow model session
        '''
        if self._session:
            self._session.close()

    def process(self, im_batch):
        '''
        - Arguments:
            - im (np.array): (batch, h, w, 3)
        
        - Returns:
            - feature_vector: (batch, f)
        '''
        im_batch = [cv2.resize(im, (self._image_shape[1], self._image_shape[0])) for im in im_batch]
        out = np.zeros((len(im_batch), self._feature_dim), np.float32)
        _run_in_batches(
            lambda x: self._session.run(self._output_var, feed_dict = x),
            {self._input_var: im_batch}, 
            out,
            self._batch_size
        )
        return out