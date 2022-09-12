# libraries
import os
import pickle
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.filters import median

# defining paths
path_to_new_frames = '/Volumes/tki/EngelhardtGroup/Saskia/data/exported_frames'
main_dir_path = os.path.join(path_to_new_frames, "flow/single_z")
path_to_res = '/Users/saskia/unibe19/master_thesis/TKI_project/scripts_fluo_data/med_filters'

med_filt = []

for root, subdirs, files in os.walk(main_dir_path):
    for name in files:
        if ('c001' in name):
            path = os.path.join(root, name)
            img = Image.open(path)

            # convert image to numpy array
            data = np.asarray(img)

            # downscale by a factor 4
            img4 = cv2.resize(data, None, fx = 1/4, fy = 1/4)

            # disky median blurr
            med = median(img4, disk(25))

            # upscale by a factor 4
            imed = cv2.resize(med, None, fx = 4, fy = 4)

            med_filt.append(imed)

# transform the list to a numpy array
med_filt = np.array(med_filt)

# average the 'list'
med_filt = np.average(med_filt, axis = 0)

# pickle the med filters somewhere
pickle.dump(med_filt, open(os.path.join(path_to_res, 'med_filt_c1.pckl'), 'wb'))
