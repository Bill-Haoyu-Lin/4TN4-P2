import numpy as np
from PIL import Image
import glob

def rgb_to_bayer(image):
    R = image[::2, ::2, 0]
    G1 = image[::2, 1::2, 1]
    G2 = image[1::2, ::2, 1]
    B = image[1::2, 1::2, 2]
    RGGB = np.zeros(image.shape[:2], dtype=np.uint8)
    RGGB[::2, ::2] = R
    RGGB[::2, 1::2] = G1
    RGGB[1::2, ::2] = G2
    RGGB[1::2, 1::2] = B
    return RGGB

def extract_patches_and_gt(image, bayer_image):
    patches =[]
    gt_data=[]
    for i in range(1, bayer_image.shape[0] - 1):
        for j in range(1, bayer_image.shape[1] - 1):
            # Extract a 5x5 patch centered at (i, j)
            patch = bayer_image[i-1:i+2, j-1:j+2]
            # change patch to 1x25
            patch = patch.flatten()
            
            # The ground truth for the missing channels for the center pixel
            # gt = image[i, j, [1, 2]] if (i + j) % 2 == 0 else image[i, j, [0, 2]]

            # if i % 2 == 0  and j%2==0: 
            #     gt = image[i, j,[1,2]]#r
            # elif i % 2 == 1 and j%2==0:
            #     gt = image[i, j,[0,2]]#g
            # elif i % 2 == 0 and j%2==1:
            #     gt = image[i, j,[0,2]]#g
            # else:
            #     gt = image[i, j,[0,1]]#b
            gt = image[i, j,0:3]
            patches.append(patch)
            gt_data.append(gt)
    print(np.array(gt_data).shape)
    return np.array(patches), np.array(gt_data)


def prepare_data(img_dir):
    img_files = glob.glob(f'{img_dir}/*.png')
    patches, gt_data = [],[]
    for img_file in img_files:
        img = np.array(Image.open(img_file))
        print(img.shape,img_file)
        bayer_img = rgb_to_bayer(img)
        p, gt = extract_patches_and_gt(img, bayer_img)
        patches.extend(p)
        gt_data.extend(gt)
    return np.array(patches), np.array(gt_data)
