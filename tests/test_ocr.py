from core.ocr import OCR_REGISTRY

def test_default_ocr():
    ocr = OCR_REGISTRY.get("DefaultOCR")(
        use_angle_cls=True,
        lang="en"
    )
    print(ocr)
