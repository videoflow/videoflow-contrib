'''
Tests that resources are present in repo releases and
can be downloaded
'''

import pytest

from videoflow.utils.downloader import get_file
from videoflow_contrib.processors.detector_tf.tf_object_detector import TensorflowObjectDetector, BASE_URL_DETECTION

def test_detector_resources():
    for modelid in TensorflowObjectDetector.supported_models:
        filename = f'{modelid}.pb'
        url_path = BASE_URL_DETECTION + filename
        get_file(filename, url_path)

if __name__ == "__main__":
    pytest.main([__file__])