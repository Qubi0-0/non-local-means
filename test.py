import cv2 as cv
import numpy as np
import matplotlib 
from numba import jit


def get_img(file_path):
    img = cv.imread(file_path)
    cv.imshow('original',img)
    return img



# img = get_img("photos/lena.png")

# image = cv.copyMakeBorder(img, 1000, 1000, 50, 50, cv.BORDER_REFLECT)
# cv.imshow('zdjecie',image)
# cv.waitKey(0)
# cv.destroyAllWindows()

for i in range(3):
    print(i)