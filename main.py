import cv2 as cv
import numpy as np
import matplotlib 


class Denoising:

    def __init__(self):
        self.img = cv.Mat


    def get_img(self,file_path):
        self.img = cv.imread(file_path)
        cv.imshow('original',self.img)

    def addNoise(self, noiseType, p = 0.001, mean = 0,  sigma = 0.3):
        ''' 
        This function takes an self.img and returns an self.img that has been noised with the given input parameters.
        p - Probability threshold of salt and pepper noise.
        noisetype - 
        '''
        if noiseType.upper() == 'GAUSSIAN':
            sigma *= 255 #Since the self.img itself is not normalized
            noise = np.zeros_like(self.img)
            noise = cv.randn(noise, mean, sigma)
            output = cv.add(self.img, noise) #generate and add gaussian noise
            return output
        elif noiseType.upper() == 'SALTNPEPPER':
            output = self.img.copy()
            noise = np.random.rand(self.img.shape[0], self.img.shape[1])
            output[noise < p] = 0
            output[noise > (1-p)] = 255
            return output

    


if __name__ == '__main__':

    denoiser = Denoising()
    denoiser.get_img("photos/lena.png")
    result = denoiser.addNoise("Gaussian", mean= 2, sigma= 4)
    cv.imshow('test',result)

    cv.waitKey(0) 
    cv.destroyAllWindows() 