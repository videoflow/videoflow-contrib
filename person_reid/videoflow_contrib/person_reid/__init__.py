from contextlib import suppress

with suppress(ImportError):
    from .mxnet_person_feature_extractor import MxnetPersonFeatureExtractor
    from .mxnet_person_reid import MxnetPersonReid
