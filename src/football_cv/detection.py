"""Adapt COCO person detections to application records; no team inference here."""
import logging
from .config import Config
from .types import Detection

log = logging.getLogger(__name__)

def resolve_device(request: str) -> str:
    """Auto-fallback to CPU, but fail clearly for an unavailable explicit GPU."""
    import torch
    if request == 'auto':
        return '0' if torch.cuda.is_available() else 'cpu'
    if request == 'cpu':
        return request
    # Ultralytics accepts a numeric device string for a single CUDA device.
    value = request.removeprefix('cuda:')
    if not value.isdigit():
        raise ValueError('device must be auto, cpu, an index such as 0, or cuda:0')
    if not torch.cuda.is_available() or int(value) >= torch.cuda.device_count():
        build = 'CPU-only PyTorch build' if torch.version.cuda is None else f'PyTorch CUDA {torch.version.cuda}'
        raise ValueError(f'Requested CUDA device {request} is unavailable ({build}); install a compatible CUDA build or use cpu/auto')
    return value


def device_description(device: str) -> str:
    """Name the actual selected hardware so a CPU fallback is visible at startup."""
    import torch
    if device == 'cpu':
        return f'CPU (PyTorch {torch.__version__}, CUDA build: {torch.version.cuda})'
    return f'CUDA:{device} {torch.cuda.get_device_name(int(device))} (PyTorch {torch.__version__}, CUDA {torch.version.cuda})'

class PersonDetector:
    """Keep model-specific tensors and class conventions behind one adapter."""
    def __init__(self, config: Config):
        from ultralytics import YOLO
        self.config = config
        self.device = resolve_device(config.device)
        log.info('Inference hardware: %s', device_description(self.device))
        self.model = YOLO(config.model)
        if self.model.task != 'detect' or self.model.names.get(0) != 'person':
            raise ValueError('Model must be a detection model with person at class 0')

    def detect(self, image) -> list[Detection]:
        """Return measured boxes in input-frame pixels, including recovery candidates."""
        # Do not apply the display threshold here: ByteTrack needs low-score boxes.
        result = self.model.predict(image, classes=[0],
            conf=min(self.config.confidence, self.config.tracking.track_low_thresh),
            device=self.device, imgsz=self.config.imgsz, iou=self.config.nms_iou,
            verbose=False)[0]
        if result.boxes is None:
            return []
        # Copy tensor values to plain records so downstream code need not use Torch.
        return [Detection(tuple(map(float, box)), float(score), int(cls))
            for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()) if int(cls) == 0]
