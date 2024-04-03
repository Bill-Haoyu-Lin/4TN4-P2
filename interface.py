import numpy as np
import torch
from PIL import Image
from model import DemosaicNet1D

GivenMosaicImg = False
Img_path = './test_img_2.png'

def load_trained_model(model_path):
    model = DemosaicNet1D()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_bayer_image(bayer_image):
    # Assume bayer_image is a numpy array representing the Bayer image
    patches = []
    for i in range(2, bayer_image.shape[0] - 2):
        for j in range(2, bayer_image.shape[1] - 2):
            patch = bayer_image[i-2:i+3, j-2:j+3]
            patch = patch.flatten()
            patches.append(patch)
    return np.array(patches)

def demosaic(model, bayer_image):
    patches = preprocess_bayer_image(bayer_image)
    patches_tensor = torch.tensor(patches, dtype=torch.float) # [N, 1, 5, 5]
    print(patches_tensor.shape)
    reconstructed_image = np.zeros((bayer_image.shape[0], bayer_image.shape[1], 3), dtype=np.uint8)
    idx = 0
    for i in range(2, bayer_image.shape[0] - 2):
        for j in range(2, bayer_image.shape[1] - 2):
            patch_pred = model(patches_tensor[idx].unsqueeze(0))  # Add batch dimension
            result = patch_pred.detach().cpu().numpy()
            # print(type(result[0]))
            outputs_np_clipped = np.clip(result, 0, 255)
            reconstructed_image[i, j, :] = outputs_np_clipped.astype(np.uint8)
            # if reconstructed_image[i,j,0]<0:
            #     print("wrong")
            # fill 2 missing channel color
            if i % 2 == 0  and j%2==0: 
                reconstructed_image[i,j,0]= patches_tensor[idx].unsqueeze(0)[0,12]
            elif i % 2 == 1 and j%2==0:
                reconstructed_image[i,j,1]= patches_tensor[idx].unsqueeze(0)[0,12]
            elif i % 2 == 0 and j%2==1:
                reconstructed_image[i,j,1]= patches_tensor[idx].unsqueeze(0)[0,12]
            else:
                reconstructed_image[i,j,2]= patches_tensor[idx].unsqueeze(0)[0,12]

            # if reconstructed_image[i, j, 1] <0 :
            #     reconstructed_image[i, j, 1]= 0
            # elif reconstructed_image[i, j, 0] < 0 :
            #     reconstructed_image[i, j, 0]= 0
            # elif reconstructed_image[i, j, 2] <0:
            #     reconstructed_image[i, j, 2]= 0


            idx += 1
    # Fill in the known channel for each pixel from the Bayer pattern
    # (This step might need adjustment based on the Bayer pattern used and the channels predicted)
    return reconstructed_image[2:-2,2:-2]

import numpy as np
from PIL import Image

def rgb_to_bayer(rgb_image):
    """
    Convert an RGB image to a Bayer pattern encoded image (RGGB pattern).

    Parameters:
    - rgb_image: A numpy array of the RGB image.

    Returns:
    - bayer_image: A numpy array representing the Bayer pattern encoded image.
    """
    # Initialize the Bayer image with zeros
    bayer_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # Red channel
    bayer_image[::2, ::2] = rgb_image[::2, ::2, 0]  # R

    # Green channel (two positions in RGGB)
    bayer_image[::2, 1::2] = rgb_image[::2, 1::2, 1]  # G
    bayer_image[1::2, ::2] = rgb_image[1::2, ::2, 1]  # G

    # Blue channel
    bayer_image[1::2, 1::2] = rgb_image[1::2, 1::2, 2]  # B

    return bayer_image


if not GivenMosaicImg:
    img_path = './img/test_img_2.png'
    rgb_img = np.array(Image.open(img_path))

    # Convert the RGB image to a Bayer pattern encoded image
    bayer_img = rgb_to_bayer(rgb_img)

    # Optionally, save the Bayer pattern image (as a grayscale image for visualization)
    Image.fromarray(bayer_img).save('./test_img_2.png')

    print("bayer img saved")

# Example usage
bayer_img = np.array(Image.open(Img_path))
model = load_trained_model('best_cnn.pth')
rgb_img = demosaic(model, bayer_img)
Image.fromarray(rgb_img).save('demosaiced_image2.png')