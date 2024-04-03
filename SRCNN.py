from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy
import math


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=(9,9),
                     activation='relu', padding="valid" , use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64,kernel_size=(3,3),
                     activation='relu', padding="same", use_bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(filters=1, kernel_size=(5,5),
                     activation='linear', padding="valid" , use_bias=True))
    adam = Adam(learning_rate=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict():
    srcnn_model = predict_model()
    srcnn_model.load_weights("3051crop_weight_200.h5")
    IMG_NAME = "./demosaiced_image.png"
    INPUT_NAME = "input2.png"
    OUTPUT_NAME = "pre2.png"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    if img.shape[1]%2 != 0:
        img = img[:,0:-1,:]

    if img.shape[1]%2 != 0:
        img = img[0:-1,:,:]

    shape = img.shape    
    print(shape)
    
    Y_img = cv2.resize(img[:, :, 0], (shape[1]*2, shape[0]*2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1]//2, shape[0]//2), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    print(im3.shape)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print ("bicubic:")
    print (cv2.PSNR(im1, im2))
    print ("SRCNN:")
    print (cv2.PSNR(im1, im3))


if __name__ == "__main__":
    predict()