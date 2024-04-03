import cv2
import numpy as np

def automatic_white_balance_hue(img_path, output_path):
    """
    Perform automatic white balance using the HSV color space and histogram equalization.
    
    Parameters:
    - img_path: Path to the input image.
    - output_path: Path where the output image will be saved.
    """
    # Load the image
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Optionally adjust the Hue here if needed

    # Apply histogram equalization to the Value channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # Convert back to RGB (BGR in OpenCV) and save or display the result
    img_bgr_balanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    img_lab= cv2.cvtColor(img_bgr_balanced, cv2.COLOR_BGR2LAB)
    
    scale_factor = 0.85 # Example scale factor: reduce brightness to 80%
    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0] * scale_factor, 0, 255)
    
    # Convert back to BGR color space from LAB
    img_bgr_balanced = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    
    # Convert back from LAB to BGR color space
    # result_img_array = cv2.cvtColor(img_bgr_balanced, cv2.COLOR_LAB2BGR)
    

    cv2.imwrite(output_path, img_bgr_balanced)

    # If you want to display the result using OpenCV
    # cv2.imshow('White Balanced Image', img_bgr_balanced)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
automatic_white_balance_hue('./img/0900x4.png', './AWB.png')
