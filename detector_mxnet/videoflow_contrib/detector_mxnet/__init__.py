from contextlib import suppress

with suppress(ImportError):
    from .mxnet_object_detector import MxnetObjectDetector
