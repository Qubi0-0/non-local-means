import cv2 as cv
import numpy as np
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt



def get_img(file_path, show = False):
    img = cv.imread(file_path)
    if show:
        cv.imshow('original',img)
    return img

def add_noise(img,  p = 0.001, mean = 0,  sigma = 0.3,show = False):
    ''' 
    This function takes an self.img and returns an img that has been noised with the given input parameters.
    p - Probability threshold of salt and pepper noise.
    '''
    sigma *= 255 #Since the img itself is not normalized
    noise = np.zeros_like(img)
    noise = cv.randn(noise, mean, sigma)
    result_g = cv.add(img, noise) #generate and add gaussian noise

    result_sp = img.copy()
    noise = np.random.rand(img.shape[0], img.shape[1])
    result_sp[noise < p] = 0
    result_sp[noise > (1-p)] = 255
    if show:
        cv.imshow('noised',result_sp)

    return result_sp , result_g


def calc_psnr(original, noisy, peak=100):
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
                        weight = np.exp(-max(euclidean_dist -2*sigma**2, 0.0)/h**2)
                        weight_sum += weight                
                        pix_val += weight*bordered_img[patch_x + patch_width,
                                                    patch_y + patch_width,layer]

                        progress += 1
                        percent_completed = progress*100/max_progress
                        if percent_completed % 5 == 0:
                            print('Completed in: ', percent_completed,'precent of layer number: ',layer+1)
                
                pix_val /= weight_sum

                result[img_x,img_y,layer] = pix_val


    return result[neighbour_width:neighbour_width+img.shape[0],neighbour_width:neighbour_width+img.shape[1]]

def mse(imageA, imageB):
    	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def non_local_means_initiate(input, neighbour_window_size, patch_window_size,h,sigma):
    # reflects borders to allow computing on the edges

    bordered_img = cv.copyMakeBorder(input, neighbour_window_size//2, neighbour_window_size//2, neighbour_window_size//2, neighbour_window_size//2, cv.BORDER_REFLECT)
    if len(input.shape) > 2:
        layers = input.shape[2]
    else: 
        layers = 1
        input = np.expand_dims(input, axis=2)
        bordered_img = np.expand_dims(bordered_img, axis=2)

    result = non_local_means_computing(input, bordered_img, neighbour_window_size, patch_window_size,sigma,h,layers)
    return result    


if __name__ == '__main__':
    x_size = 2
    y_size = 2
    photo_num = [1,3] 
    for num in tqdm(photo_num):
        img = get_img(f"photos/gray/{num}.bmp")
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sp_noised, g_noised = add_noise(img, p= 0.05, mean= 0, sigma= 0.15)
        result = non_local_means_initiate(sp_noised,neighbour_window_size= 30,patch_window_size= 10,h = 0.45*64,sigma= 64) # neighbour_window_size= 20,patch_window_size= 6,h = 18,sigma= 38
        plt.figure(figsize=(18,10))
        plt.axis("off")
        plt.subplot(x_size,y_size,1)
        plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        plt.xlabel("Original")
        plt.subplot(x_size,y_size,2)
        plt.imshow(cv.cvtColor(sp_noised,cv.COLOR_BGR2RGB))
        plt.title("PSNR {0:.2f}dB".format(calc_psnr(img, sp_noised),"MSE: {0:.2f}".format(mse(img, result))))
        plt.xlabel("Salt&Pepper")
        plt.subplot(x_size,y_size,3)
        plt.imshow(cv.cvtColor(result,cv.COLOR_BGR2RGB))
        plt.title("PSNR {0:.2f}dB".format(calc_psnr(img, result),"MSE: {0:.2f}".format(mse(img, result))))
        plt.xlabel("Denoised")
        plt.subplot(x_size,y_size,4)
        plt.imshow(cv.cvtColor(cv.subtract(img,result),cv.COLOR_BAYER_BG2GRAY))
        plt.xlabel("Difference: Orginal - Densoised")
        plt.show()
    cv.waitKey(0) 
    cv.destroyAllWindows()