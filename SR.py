import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('./geeks.png')

# Read the desired model
path = "./EDSR_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./upscaled.png", result)

result1 = cv2.resize(result,[result.shape[1]//2,result.shape[0]//2],cv2.INTER_CUBIC)

cv2.imwrite("./UnD.png", result1)