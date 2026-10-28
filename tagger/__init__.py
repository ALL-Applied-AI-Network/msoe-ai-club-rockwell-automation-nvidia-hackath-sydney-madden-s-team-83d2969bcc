"""Automated Video Tagger Package.

Zero-lookup computer vision engine for extracting discrete dispense stream timestamps,
fill volumes, and vial station cycles from Rockwell Automation Station 3 video recordings.
"""

__all__ = ["AutoTagger", "VolumeEstimator"]


def __getattr__(name):
    if name == "AutoTagger":
        from .auto_tagger import AutoTagger
        return AutoTagger
    if name == "VolumeEstimator":
        from .volume_estimator import VolumeEstimator
        return VolumeEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
