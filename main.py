import cv2 as cv
import numpy as np
import matplotlib 
from numba import jit


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

# @staticmethod
@jit(nopython=True) 
def non_local_means(input, neighbour_window_size, patch_window_size,h):
    '''
    Performs the non-local-means algorithm given a input img.
    params is a tuple with:
    params = (neighbour_window_size, patch_window_size, h)
    Please keep neighbour_window_size and patch_window_size as even numbers
    '''

        

    pad_width = neighbour_window_size//2
    img = input.copy()
    # The next few lines creates a padded img that reflects the border so that the big window can be accomodated through the loop - Do zmienienia ew.
    # if len(img.shape) < 3:
    padded_img = np.zeros((img.shape[0] + neighbour_window_size,img.shape[1] + neighbour_window_size))
    layers = 1
    # else: 
        # padded_img = np.zeros((img.shape[0] + neighbour_window_size,img.shape[1] + neighbour_window_size,img.shape[2]))
        # layers = 3
    padded_img = padded_img.astype(np.uint8)
    padded_img[pad_width:pad_width+img.shape[0], pad_width:pad_width+img.shape[1]] = img
    padded_img[pad_width:pad_width+img.shape[0], 0:pad_width] = np.fliplr(img[:,0:pad_width])
    padded_img[pad_width:pad_width+img.shape[0], img.shape[1]+pad_width:img.shape[1]+2*pad_width] = np.fliplr(img[:,img.shape[1]-pad_width:img.shape[1]])
    padded_img[0:pad_width,:] = np.flipud(padded_img[pad_width:2*pad_width,:])
    padded_img[pad_width+img.shape[0]:2*pad_width+img.shape[0], :] =np.flipud(padded_img[padded_img.shape[0] - 2*pad_width:padded_img.shape[0] - pad_width,:])
    max_progress = img.shape[1]*img.shape[0]*(neighbour_window_size - patch_window_size)**2
    result = padded_img.copy()
    for layer in range(layers):
        progress = 0
        for img_x in range(pad_width , pad_width + img.shape[0]):
            for img_y in range(pad_width, pad_width + img.shape[1]):
                win_x = img_x - pad_width
                win_y = img_y - pad_width
                # if layers == 3:
                    # neighbour_window = padded_img[img_x - patch_window_size//2 : img_x + patch_window_size//2 +1,
                                                    # img_y - patch_window_size//2 : img_y + patch_window_size//2 +1,layer]
                # else:
                neighbour_window = padded_img[img_x - patch_window_size//2 : img_x + patch_window_size//2 +1,
                                                img_y - patch_window_size//2 : img_y + patch_window_size//2 +1]

                pix_val = 0
                weight_sum = 0

                for patch_x in range(win_x,win_x + neighbour_window_size - patch_window_size):
                    for patch_y in range(win_y,win_y + neighbour_window_size - patch_window_size):
                        # if layers == 3:
                        # patch_window = padded_img[patch_x:patch_x+patch_window_size + 1,
                                                    #   patch_y:patch_y+patch_window_size + 1,layer]
                        # else:                        
                        patch_window = padded_img[patch_x:patch_x+patch_window_size + 1,
                                                      patch_y:patch_y+patch_window_size + 1]

                        euclidean_dist = np.sqrt(np.sum(np.square(patch_window - neighbour_window)))
                        weight = np.exp(-euclidean_dist/h)
                        weight_sum += weight 
                        # if layers == 3:
                        # pix_val += weight*padded_img[patch_x + patch_window_size//2,
                                                        #  patch_y + patch_window_size//2,layer]
                        # else: 
                        pix_val += weight*padded_img[patch_x + patch_window_size//2,
                                                         patch_y + patch_window_size//2]    
                        progress += 1
                        percent_completed = progress*100/max_progress
                        if percent_completed % 5 == 0:
                            # print(f'COMPLETED = {percent_completed} percent of layer number: {layer +1}')
                                print('Completed : ', percent_completed)
                                print('of layer nubmer: ',layer+1)
                pix_val /= weight_sum
                if layers == 3:
                    result[img_x,img_y,layer] = pix_val
                else:
                    result[img_x,img_y] = pix_val 


    # cv.imshow("Final result",result)
    return result[pad_width:pad_width+img.shape[0],pad_width:pad_width+img.shape[1]]

@jit(nopython=True)
def nonLocalMeans(input, params = tuple(), verbose = True):
  '''
  Performs the non-local-means algorithm given a input image.
  params is a tuple with:
  params = (bigWindowSize, smallWindowSize, h)
  Please keep bigWindowSize and smallWindowSize as even numbers
  '''
              

  bigWindowSize, smallWindowSize, h  = params
  padwidth = bigWindowSize//2
  image = input.copy()

  # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
  paddedImage = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
  paddedImage = paddedImage.astype(np.uint8)
  paddedImage[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = image
  paddedImage[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(image[:,0:padwidth])
  paddedImage[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(image[:,image.shape[1]-padwidth:image.shape[1]])
  paddedImage[0:padwidth,:] = np.flipud(paddedImage[padwidth:2*padwidth,:])
  paddedImage[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImage[paddedImage.shape[0] - 2*padwidth:paddedImage.shape[0] - padwidth,:])
  


  iterator = 0
  totalIterations = image.shape[1]*image.shape[0]*(bigWindowSize - smallWindowSize)**2

  if verbose:
    print("TOTAL ITERATIONS = ", totalIterations)

  outputImage = paddedImage.copy()

  smallhalfwidth = smallWindowSize//2


  # For each pixel in the actual image, find a area around the pixel that needs to be compared
  for imageX in range(padwidth, padwidth + image.shape[1]):
    for imageY in range(padwidth, padwidth + image.shape[0]):
      
      bWinX = imageX - padwidth
      bWinY = imageY - padwidth

      #comparison neighbourhood
      compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,
                             imageX - smallhalfwidth:imageX + smallhalfwidth + 1]
      
      
      pixelColor = 0
      totalWeight = 0

      # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
      for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
        for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):   
          #find the small box       
          smallNbhd = paddedImage[sWinY:sWinY+smallWindowSize + 1,
                                    sWinX:sWinX+smallWindowSize + 1]
          euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
          #weight is computed as a weighted softmax over the euclidean distances
          weight = np.exp(-euclideanDistance/h)
          totalWeight += weight
          pixelColor += weight*paddedImage[sWinY + smallhalfwidth,
                                             sWinX + smallhalfwidth]
          iterator += 1

          if verbose:
            percentComplete = iterator*100/totalIterations
            if percentComplete % 5 == 0:
              print('% COMPLETE = ', percentComplete)

      pixelColor /= totalWeight
      outputImage[imageY, imageX] = pixelColor

  return outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]


if __name__ == '__main__':

    lena_img = get_img("photos/lena.png")
    grey = cv.cvtColor(lena_img, cv.COLOR_BGR2GRAY) 
    result = add_noise(grey,"SALTNPEPPER", p= 0.001, mean= 0, sigma= 0.3)
    cv.imshow('noised',result)
    result_final = non_local_means(result,20,6,16)
    cv.imshow("Final", result_final)
    dupa = (20,6,16)
     
    result_final_2 = nonLocalMeans(result,dupa)
    cv.imshow("Final_2", result_final_2)
    cv.waitKey(0) 
    cv.destroyAllWindows()