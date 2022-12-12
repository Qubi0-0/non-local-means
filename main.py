import pandas as pd
import cv2 as cv
import numpy as np
from time import sleep
from tqdm import tqdm
import non_local_means as nlm
from pathlib import Path
import os
import csv
def data_collector(path,data):
    pass





if __name__ == '__main__':

    sp_values = [0.001, 0.002, 0.005, 0.010, 0.015]
    gauss_values_mean = [0 , 0.1, 0.2 , 0.3, 0.5]
    gauss_values_sigma = [0 , 0.1, 0.2 , 0.3, 0.5]
    h_values = []
    sigma_values = []
    neighbour_window_size_values = []
    patch_window_size_values = []



    cv_result = cv.Mat
    file = open('data.csv', 'w')
    writer = csv.writer(file)
    header = ["Index", "Noise Type", "Metric", "Noised", "NLM from OpenCV", "Our NLM"]
    writer.writerow(header)
    img_num = 0
    grey_folder = Path('photos/gray')
    color_folder = Path('photos/color')
    for i in tqdm(range(len(sp_values))):
        type = ['Salt&Pepper', "Gaussian"]
        for filename in (os.listdir(grey_folder)):
            img_num += 1
            f = os.path.join(grey_folder, filename)
            # checking if it is a file
            if os.path.isfile(f):
                img = cv.imread(f)
                sp_noised, g_noised = nlm.add_noise(img,sp_values[i], gauss_values_mean[i],gauss_values_sigma[i])
                psnr_noised_sp = nlm.calc_psnr(img,sp_noised)
                psnr_noised_g = nlm.calc_psnr(img,g_noised)
                cv_result = cv.fastNlMeansDenoising(sp_noised,None,15,6,20)
                cv_psnr = nlm.calc_psnr(img,cv_result)
                my_result = nlm.non_local_means_initiate(sp_noised,neighbour_window_size= 20,patch_window_size= 6,h = 21,sigma= 40)
                my_psnr = nlm.calc_psnr(img,my_result)
                data = [img_num+1,type[0],"PSNR",psnr_noised_sp,cv_psnr,my_psnr]
                writer.writerow(data)
                cv_result = cv.fastNlMeansDenoising(sp_noised,None,15,6,20)
                cv_psnr = nlm.calc_psnr(img,cv_result)
                my_result = nlm.non_local_means_initiate(sp_noised,neighbour_window_size= 20,patch_window_size= 6,h = 21,sigma= 40)
                my_psnr = nlm.calc_psnr(img,my_result)
                data = [img_num+1,type[1],"PSNR",psnr_noised_g,cv_psnr,my_psnr]
                writer.writerow(data)