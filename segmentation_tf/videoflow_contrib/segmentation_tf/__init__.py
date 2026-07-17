from contextlib import suppress

with suppress(ImportError):
    from .tf_segmentation import BASE_URL_SEGMENTATION, TensorflowSegmenter
