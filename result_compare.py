from PIL import Image
import numpy as np

def calculate_mse_central_region(original_image, demosaiced_image):
    """
    Calculate the Mean Squared Error between the central region of the original image
    and the demosaiced image.
    
    Parameters:
    - original_image: numpy array of the original image.
    - demosaiced_image: numpy array of the demosaiced image.
    
    Returns:
    - mse: Mean Squared Error as a float.
    """
    # Calculate the amount to crop from each side of the original image
    crop_size = 1 # Number of pixels to crop from each side
    
    # Assuming original_image and demosaiced_image are numpy arrays
    # Crop the original image to match the size of the demosaiced image
    cropped_original = original_image[crop_size:-crop_size, crop_size:-crop_size]

    cropped_original = original_image
    
    print(cropped_original.shape,demosaiced_image.shape)
    # Ensure the cropped original and demosaiced images have the same shape
    if cropped_original.shape != demosaiced_image.shape:
        raise ValueError("Cropped original image and demosaiced image must have the same dimensions.")
    
    
    # Calculate MSE
    mse = np.mean((cropped_original - demosaiced_image) ** 2)
    return mse

def load_image_as_array(image_path):
    """
    Load an image from the given path and convert it to a numpy array.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - Numpy array of the image.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img)
    return img_array

# Continue using the load_image_as_array function from the previous example to load images

# Example paths to your images
original_image_path = './img/0900x4.png'
demosaiced_image_path = './geeks.png'

# Load images as numpy arrays
original_image = load_image_as_array(original_image_path)
demosaiced_image = load_image_as_array(demosaiced_image_path)

# Calculate MSE for the central regions
mse = calculate_mse_central_region(original_image, demosaiced_image)
print(f"Mean Squared Error (Central Region): {mse}")