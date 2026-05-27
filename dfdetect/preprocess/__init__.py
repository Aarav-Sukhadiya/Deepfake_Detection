"""Face detection and frame sampling."""
from .face_detect import FaceDetector, build_detector
from .frame_sampler import sample_image, sample_video

__all__ = ["FaceDetector", "build_detector", "sample_image", "sample_video"]
