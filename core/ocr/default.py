import numpy as np
from paddleocr import PaddleOCR
from . import OCR_REGISTRY


@OCR_REGISTRY.register()
class DefaultOCR:
    """
    Default OCR model
    """
    def __init__(self, use_angle_cls=True, lang='en') -> None:
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
    def forward(self, image : np.ndarray) -> str:
        res = self.ocr.ocr(image, cls=True)[0]
        doc = ' '.join([l[-1][0] for l in res])
        return doc