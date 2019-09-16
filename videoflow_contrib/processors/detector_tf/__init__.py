from contextlib import suppress

with suppress(ImportError):
    from .tf_object_detector import TensorflowObjectDetector, BASE_URL_DETECTION