from .config import Config
from .types import Detection

def resolve_device(request: str) -> str:
    import torch
    if request == 'auto':
        return '0' if torch.cuda.is_available() else 'cpu'
    if request == 'cpu':
        return request
    value = request.removeprefix('cuda:')
    if not value.isdigit():
        raise ValueError('device must be auto, cpu, an index such as 0, or cuda:0')
    if not torch.cuda.is_available() or int(value) >= torch.cuda.device_count():
        raise ValueError(f'Requested CUDA device {request} is unavailable; use cpu or auto')
    return value

class PersonDetector:
    def __init__(self, config: Config):
        from ultralytics import YOLO
        self.config = config
        self.device = resolve_device(config.device)
        self.model = YOLO(config.model)
        if self.model.task != 'detect' or self.model.names.get(0) != 'person':
            raise ValueError('Model must be a detection model with person at class 0')

    def detect(self, image) -> list[Detection]:
        result = self.model.predict(image, classes=[0],
            conf=min(self.config.confidence, self.config.tracking.track_low_thresh),
            device=self.device, imgsz=self.config.imgsz, iou=self.config.nms_iou,
            verbose=False)[0]
        if result.boxes is None:
            return []
        return [Detection(tuple(map(float, box)), float(score), int(cls))
            for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()) if int(cls) == 0]
