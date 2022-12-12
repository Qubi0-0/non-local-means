import pandas as pd
import cv2 as cv
import numpy as np
from time import sleep
from tqdm import tqdm
import non_local_means as nlm
from pathlib import Path
import os

def data_collector(path,data):
    pass





if __name__ == '__main__':

    sp_values = [0.001, 0.002, 0.005, 0.010, 0.015]
    gauss_values_mean = [0 , 0.1, 0.2 , 0.3, 0.5]
    gauss_values_sigma = [0 , 0.1, 0.2 , 0.3, 0.5]



    grey_folder = Path('photos/gray')
    color_folder = Path('photos/color')

    for filename in tqdm(os.listdir(grey_folder)):
        f = os.path.join(grey_folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            img = cv.imread(f)
            sp_noised = nlm.add_noise(img, "SP")
