from contextlib import suppress

with suppress(ImportError):
    from .sort import KalmanFilterBoundingBoxTracker