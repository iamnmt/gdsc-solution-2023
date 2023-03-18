from . import PREPROCESS_REGISTRY

import cv2
import numpy as np 

@PREPROCESS_REGISTRY.register()
class DefaultPreprocessor:
    """
    Default preprocessor for the OCR model.
    
    Converts the image to grayscale, removes noise, 
    and crops the image to the biggest contour.
    """
    def __init__(self) -> None:
        pass
    def forward(self, image: np.ndarray) -> np.ndarray:
        assert len(image.shape) == 3, "Image must be 3D"
        assert image.shape[2] == 3, "Image must be RGB"
        assert image.dtype == np.uint8, "Image must be uint8"

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)

        # Crop the image
        # Find all contours in the image.
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Retreive the biggest contour
        biggest_contour = max(contours, key = cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(biggest_contour)

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        crop_image = image[y:y+h, x:x+w]
        
        return crop_image