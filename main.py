import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from time import sleep
from tqdm import tqdm
import random

def get_img(file_path, show = False):
    img = cv.imread(file_path)
    if show:
        cv.imshow('original',img)
    return img

def add_noise(img,noiseType, p = 0.001, mean = 0,  sigma = 0.3, show = False):
    ''' 
    This function takes an self.img and returns an self.img that has been noised with the given input parameters.
    p - Probability threshold of salt and pepper noise.
    noisetype - 
    '''
    if noiseType.upper() == 'GAUSSIAN':
        sigma *= 255 #Since the img itself is not normalized
        noise = np.zeros_like(img)
        noise = cv.randn(noise, mean, sigma)
        result = cv.add(img, noise) #generate and add gaussian noise
    elif noiseType.upper() == 'SALTNPEPPER':
        result = img.copy()
        noise = np.random.rand(img.shape[0], img.shape[1])
        result[noise < p] = 0
        result[noise > (1-p)] = 255
    if show:
        cv.imshow('noised',result)
    return result


def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)


@jit(nopython=True) 
def non_local_means_computing(input, bordered_img, neighbour_window_size, patch_window_size, sigma,h ,layers):
    '''Performs the non-local-means algorithm given a input img.'''

    neighbour_width = neighbour_window_size//2
    patch_width = patch_window_size//2
    img = input.copy()
    max_progress = (img.shape[1]*img.shape[0]*(neighbour_window_size - patch_window_size)**2)*layers
    result = bordered_img.copy()

    

    # with tqdm(total=max_progress) as progress_bar:
    for layer in range(layers):
        progress = 0
        for img_x in range(neighbour_width , neighbour_width + img.shape[0]):
            for img_y in range(neighbour_width, neighbour_width + img.shape[1]):
                win_x = img_x - neighbour_width
                win_y = img_y - neighbour_width
                
                neighbour_window = bordered_img[img_x - patch_window_size//2 : img_x + patch_width +1,
                                                img_y - patch_window_size//2 : img_y + patch_width +1,layer]

                pix_val = 0
                weight_sum = 0
                for patch_x in range(win_x,win_x + neighbour_window_size - patch_window_size):
                    for patch_y in range(win_y,win_y + neighbour_window_size - patch_window_size):

                        patch_window = bordered_img[patch_x:patch_x+patch_window_size + 1,
                                                    patch_y:patch_y+patch_window_size + 1,layer]


                        euclidean_dist = (np.sum(np.square(patch_window - neighbour_window)))
                        # euclidean_dist = euclidean_dist/3*(patch_window_size**2)                        
                        weight = np.exp(-max(euclidean_dist -2*sigma**2, 0.0)/h**2)
                        weight_sum += weight                
                        pix_val += weight*bordered_img[patch_x + patch_width,
                                                    patch_y + patch_width,layer]
                        # progress_bar.update(1)
                        progress += 1
                        percent_completed = progress*100/max_progress
                        if percent_completed % 10 == 0:
                            print('Completed in: ', percent_completed,'precent of layer number: ',layer+1)
                
                pix_val /= weight_sum

                result[img_x,img_y,layer] = pix_val


    return result[neighbour_width:neighbour_width+img.shape[0],neighbour_width:neighbour_width+img.shape[1]]



def non_local_means_initiate(input, neighbour_window_size, patch_window_size,h,sigma):
    # reflects borders to allow computing on the edges
    bordered_img = cv.copyMakeBorder(input, neighbour_window_size//2, neighbour_window_size//2, neighbour_window_size//2, neighbour_window_size//2, cv.BORDER_REFLECT)
    if len(input.shape) > 2:
        layers = input.shape[2]
        bordered_img = bordered_img.shape[2]
    else: 
        layers = 1
        input = np.expand_dims(input, axis=2)
        bordered_img = np.expand_dims(bordered_img, axis=2)

    result = non_local_means_computing(input, bordered_img, neighbour_window_size, patch_window_size,sigma,h,layers)
    cv.imshow("Final", result)


if __name__ == '__main__':

    lena_img = get_img("photos/lena.png")
    lena_img = cv.cvtColor(lena_img, cv.COLOR_BGR2GRAY) 
    result = add_noise(lena_img,"SALTNPEPPER", p= 0.001, mean= 0, sigma= 0.3,show = True)
    non_local_means_initiate(result,neighbour_window_size= 20,patch_window_size= 6,h = 19,sigma= 36)

    cv.waitKey(0) 
    cv.destroyAllWindows()