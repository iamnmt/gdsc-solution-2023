from . import PREPROCESS

@PREPROCESS.register()

def grayscale(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def noise_removal(image: np.ndarray) -> np.ndarray :
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def find_and_crop_contours(temp: np.ndarray, image: np.ndarray) -> np.ndarray:

    temp = cv2.adaptiveThreshold(result,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 5)
    
    # Find all contours in the image.
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Sort the contours in decreasing order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
    image1_copy = image.copy()
    # Draw largest 3 contours
    for i, cont in enumerate(sorted_contours[:10],1):

        # Draw the contour
        cv2.drawContours(image1_copy, cont, -1, (0,255,0), 3)
    
        # Display the position of contour in sorted list
        cv2.putText(image1_copy, str(i), (cont[0,0,0], cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0),4)

    # Retreive the biggest contour
    biggest_contour = max(contours, key = cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(biggest_contour)

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    crop_image = image[y:y+h, x:x+w]
    
    return crop_image
