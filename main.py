import cv2 as cv
import numpy as np
import matplotlib 
from numba import jit
from time import sleep
from tqdm import tqdm
import random

def get_img(file_path):
    img = cv.imread(file_path)
    cv.imshow('original',img)
    return img

def add_noise(img,noiseType, p = 0.001, mean = 0,  sigma = 0.3):
    ''' 
    This function takes an self.img and returns an self.img that has been noised with the given input parameters.
    p - Probability threshold of salt and pepper noise.
    noisetype - 
    '''
    if noiseType.upper() == 'GAUSSIAN':
        sigma *= 255 #Since the img itself is not normalized
        noise = np.zeros_like(img)
        noise = cv.randn(noise, mean, sigma)
        output = cv.add(img, noise) #generate and add gaussian noise
        return output
    elif noiseType.upper() == 'SALTNPEPPER':
        output = img.copy()
        noise = np.random.rand(img.shape[0], img.shape[1])
        output[noise < p] = 0
        output[noise > (1-p)] = 255
        return output




@jit(nopython=True) 
def non_local_means_computing(input, neighbour_window_size, patch_window_size,h,layers):
    '''Performs the non-local-means algorithm given a input img.'''

    neighbour_width = neighbour_window_size//2
    patch_width = patch_window_size//2
    img = input.copy()
    

    bordered_img = np.zeros((img.shape[0] + neighbour_window_size,img.shape[1] + neighbour_window_size,layers))
    bordered_img = bordered_img.astype(np.uint8)
    bordered_img[neighbour_width:neighbour_width+img.shape[0], neighbour_width:neighbour_width+img.shape[1]] = img
    bordered_img[neighbour_width:neighbour_width+img.shape[0], 0:neighbour_width] = np.fliplr(img[:,0:neighbour_width])
    bordered_img[neighbour_width:neighbour_width+img.shape[0], img.shape[1]+neighbour_width:img.shape[1]+2*neighbour_width] = np.fliplr(img[:,img.shape[1]-neighbour_width:img.shape[1]])
    bordered_img[0:neighbour_width,:] = np.flipud(bordered_img[neighbour_width:2*neighbour_width,:])
    bordered_img[neighbour_width+img.shape[0]:2*neighbour_width+img.shape[0], :] =np.flipud(bordered_img[bordered_img.shape[0] - 2*neighbour_width:bordered_img.shape[0] - neighbour_width,:])
    max_progress = (img.shape[1]*img.shape[0]*(neighbour_window_size - patch_window_size)**2)
    result = bordered_img.copy()



    # with tqdm(total=max_progress) as progress_bar:
    for layer in range(layers):
        progress = 0
        for img_x in  range(neighbour_width , neighbour_width + img.shape[0]):
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


                        euclidean_dist = np.sqrt(np.sum(np.square(patch_window - neighbour_window)))
                        weight = np.exp(-euclidean_dist/h)
                        weight_sum += weight 
                
                        pix_val += weight*bordered_img[patch_x + patch_width,
                                                    patch_y + patch_width,layer]
                        # progress_bar.update(1)
                        progress += 1
                        percent_completed = progress*100/max_progress
                        if percent_completed % 10 == 0:
                            print('Completed : ', percent_completed,'precent of layer number: ',layer+1)

                pix_val /= weight_sum

                result[img_x,img_y,layer] = pix_val


    return result[neighbour_width:neighbour_width+img.shape[0],neighbour_width:neighbour_width+img.shape[1]]



def non_local_means_initiate(input, neighbour_window_size, patch_window_size,h):
    if len(input.shape) > 2:
        layers = input.shape[2]
    else: 
        layers = 1
        input = np.expand_dims(input, axis=2)
    result = non_local_means_computing(input, neighbour_window_size, patch_window_size,h,layers)
    cv.imshow("Final", result)


if __name__ == '__main__':

    lena_img = get_img("photos/lena.png")
    lena_img = cv.cvtColor(lena_img, cv.COLOR_BGR2GRAY) 
    result = add_noise(lena_img,"SALTNPEPPER", p= 0.001, mean= 0, sigma= 0.3)
    cv.imshow('noised',result)
    non_local_means_initiate(result,20,6,16)

    cv.waitKey(0) 
    cv.destroyAllWindows()