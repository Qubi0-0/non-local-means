import cv2 as cv
import numpy as np
import matplotlib 


class Denoising:

    def __init__(self):
        self.img = cv.Mat


    def get_img(self,file_path):
        self.img = cv.imread(file_path)
        cv.imshow('original',self.img)
        return self.img

    def add_noise(self, img,noiseType, p = 0.001, mean = 0,  sigma = 0.3):
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


    def non_local_means(self,noisy, big_window_size, small_window_size, verbose = True):
        '''
        Performs the non-local-means algorithm given a noisy img.
        params is a tuple with:
        params = (big_window_size, small_window_size, h)
        Please keep big_window_size and small_window_size as even numbers
        '''


        # noisy = cv.cvtColor(noisy, cv.COLOR_BGR2GRAY)                
        pad_width = big_window_size//2
        img = noisy.copy()
        progress = 0
        # The next few lines creates a padded img that reflects the border so that the big window can be accomodated through the loop - Do zmienienia ew.
        padded_img = np.zeros((img.shape[0] + big_window_size,img.shape[1] + big_window_size,img.shape[2]))
        padded_img = padded_img.astype(np.uint8)
        padded_img[pad_width:pad_width+img.shape[0], pad_width:pad_width+img.shape[1]] = img
        padded_img[pad_width:pad_width+img.shape[0], 0:pad_width] = np.fliplr(img[:,0:pad_width])
        padded_img[pad_width:pad_width+img.shape[0], img.shape[1]+pad_width:img.shape[1]+2*pad_width] = np.fliplr(img[:,img.shape[1]-pad_width:img.shape[1]])
        padded_img[0:pad_width,:] = np.flipud(padded_img[pad_width:2*pad_width,:])
        padded_img[pad_width+img.shape[0]:2*pad_width+img.shape[0], :] =np.flipud(padded_img[padded_img.shape[0] - 2*pad_width:padded_img.shape[0] - pad_width,:])

        max_progress = img.shape[1]*img.shape[0]*(big_window_size - small_window_size)**2
        result = padded_img.copy()
        cv.imshow('padded',padded_img)
        # for i in range(pad_width , pad_width + img.shape[0]):
        #     for j in range(pad_width, pad_width + img.shape[1]):


if __name__ == '__main__':

    denoiser = Denoising()
    lena_img = denoiser.get_img("photos/lena.png")
    result = denoiser.add_noise(lena_img,"Gaussian", mean= 2, sigma= 4)
    # cv.imshow('test',result)
    params = (5,9,2)
    denoiser.non_local_means(lena_img,20,10)
    cv.waitKey(0) 
    cv.destroyAllWindows()