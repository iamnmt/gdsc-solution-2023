import numpy as np
from ..preprocess import PREPROCESS_REGISTRY
from ..ocr import OCR_REGISTRY
from ..labelers import LABELER_REGISTRY


class BaseModel:
    def __init__(self, cfg) -> None:
        self.preprocessor = (
            PREPROCESS_REGISTRY.get(cfg.preprocessor["name"])(
                **cfg.preprocessor["args"]
            )
            if cfg.preprocessor["args"]
            else PREPROCESS_REGISTRY.get(cfg.preprocessor["name"])()
        )
        self.ocr = OCR_REGISTRY.get(cfg.ocr["name"])(**cfg.ocr["args"])
        self.labelr = LABELER_REGISTRY.get(cfg.labeler["name"])(**cfg.labeler["args"])

    def forward(self, image: np.ndarray) -> list[str]:
        img = self.preprocessor.forward(image)
        doc = self.ocr.forward(img)
        labels = self.labeler.forward([doc])
        return labels
