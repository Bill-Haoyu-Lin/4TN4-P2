import cv2
import numpy as np

def recolor_edges(image_path):
    # Load the image

    img = cv2.imread(image_path)

    img_original = cv2.imread('./img/0900x4.png')
    crop_size = 2 # Number of pixels to crop from each side
    
    # Assuming original_image and demosaiced_image are numpy arrays
    # Crop the original image to match the size of the demosaiced image
    cropped_original = img_original[crop_size:-crop_size, crop_size:-crop_size]
    
    img_gray = cv2.cvtColor(cropped_original , cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 120, 244)
    
    # Prepare an output image where we'll set the recolored pixels
    output_img = img.copy()
    
    # Find coordinates of edge pixels
    y_coords, x_coords = np.where(edges > 0)
    
    # Iterate over edge pixels
    for y, x in zip(y_coords, x_coords):
        # Define the 3x3 surrounding patch
        x_min, x_max = max(0, x-1), min(img.shape[1], x+2)
        y_min, y_max = max(0, y-1), min(img.shape[0], y+2)
        patch = img[y_min:y_max, x_min:x_max]
        
        # Calculate the mean color of the patch, excluding the edge pixel itself
        # This is a simple way to exclude the center pixel, by averaging all and subtracting the center
        mean_color = patch.reshape(-1, 3).mean(axis=0)
        
        # Recolor the edge pixel in the output image
        output_img[y, x] = mean_color
    
    return output_img

# Load, process, and display the image
image_path = './UnD.png'
processed_img = recolor_edges(image_path)

# Optionally, save the processed image
output_path = './AfterEdge.png'
cv2.imwrite(output_path, processed_img)