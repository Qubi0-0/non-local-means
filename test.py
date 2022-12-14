import cv2 as cv
import numpy as np
import matplotlib 
from numba import jit


def get_img(file_path):
    img = cv.imread(file_path)
    cv.imshow('original',img)
    return img

def gauss(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (512, 512)) #  np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv.normalize(noisy_image, noisy_image, 0, 255, cv.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

img = get_img(f"photos/color/1.bmp")

noised = gauss(img)
cv.imshow("noised",noised)


img = cv.imread("photos/color/1.bmp")[...,::-1]/255.0
noise =  np.random.normal(loc=0, scale=1, size=img.shape)

# noise overlaid over image
noisy = cv.add((img, noise*0.2))
cv.imshow('num2',noisy)

cv.waitKey(0)
cv.destroyAllWindows()
