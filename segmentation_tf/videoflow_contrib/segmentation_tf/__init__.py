from contextlib import suppress

with suppress(ImportError):
    from .tf_segmentation import TensorflowSegmenter, BASE_URL_SEGMENTATION