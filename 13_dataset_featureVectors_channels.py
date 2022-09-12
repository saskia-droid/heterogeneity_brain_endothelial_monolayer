# libraries
import os
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.filters import median

if __name__ == '__main__':

    # defining paths
    path_to_new_frames = '/Volumes/tki/EngelhardtGroup/Saskia/data/exported_frames'
    main_dir_path = os.path.join(path_to_new_frames, "flow/single_z")
    path_to_preprocess_channel_images = '/Users/saskia/unibe19/master_thesis/TKI_project/data/preprocess_channel_images/flow/single_z'
    path_to_dataset = '/Users/saskia/unibe19/master_thesis/TKI_project/fluo_datasets/'

    # upload the model
    models_path = '/Users/saskia/unibe19/master_thesis/TKI_project/UBELIX/models/'
    model = tf.keras.models.load_model(os.path.join(models_path, '78_3c_l2Normalize_lrelu_bestEpoch.h5'))

    # extract trained layers that we need
    model.layers.pop(0)
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.Input(shape=(None, None, 1)))  # change input dim
    for layer in model.layers[1:-2]:  # -2 -> third last layer as output
        new_model.add(layer)

    c0 = []
    c1 = []
    c2 = []
    c3 = []

    for i in range(18):
        ### c0

        img = Image.open(os.path.join(main_dir_path, str(i), 'c000_t009_z000.png'))

        # convert image to numpy array
        data = np.asarray(img)

        # get feature map
        feature_map = new_model.predict(data[None, :, :, None])

        fv = feature_map.reshape(-1, feature_map.shape[-1])
        c0.append(fv)

        ### c1

        # original image
        #img = Image.open(os.path.join(main_dir_path, str(i), 'c001_t009_z000.png'))
        # pre-process image
        img = Image.open(os.path.join(path_to_preprocess_channel_images, str(i), 'c001_t009_z000.png'))

        # convert image to numpy array
        data = np.asarray(img)

        img32 = cv2.resize(data, None, fx=1 / 32, fy=1 / 32, interpolation = cv2.INTER_AREA)
        c1.append(img32)

        ### c2

        # original image
        #img = Image.open(os.path.join(main_dir_path, str(i), 'c002_t009_z000.png'))
        # pre-process image
        img = Image.open(os.path.join(path_to_preprocess_channel_images, str(i), 'c002_t009_z000.png'))

        # convert image to numpy array
        data = np.asarray(img)

        img32 = cv2.resize(data, None, fx=1 / 32, fy=1 / 32, interpolation = cv2.INTER_AREA)
        c2.append(img32)

        ### c3

        # original image
        #img = Image.open(os.path.join(main_dir_path, str(i), 'c003_t009_z000.png'))
        # pre-process image
        img = Image.open(os.path.join(path_to_preprocess_channel_images, str(i), 'c003_t009_z000.png'))

        # convert image to numpy array
        data = np.asarray(img)

        img32 = cv2.resize(data, None, fx=1 / 32, fy=1 / 32, interpolation = cv2.INTER_AREA)
        c3.append(img32)

    # transform the list to a numpy array
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    c0 = c0.reshape(-1, 64)
    c1 = c1.reshape(-1, 1)
    c2 = c2.reshape(-1, 1)
    c3 = c3.reshape(-1, 1)

    # pickle everything...
    pickle.dump(c0, open(os.path.join(path_to_dataset, 'c0.pckl'), 'wb'))
    pickle.dump(c1, open(os.path.join(path_to_dataset, 'c1_preprocess.pckl'), 'wb'))
    pickle.dump(c2, open(os.path.join(path_to_dataset, 'c2_preprocess.pckl'), 'wb'))
    pickle.dump(c3, open(os.path.join(path_to_dataset, 'c3_preprocess.pckl'), 'wb'))

    #pickle.dump(c1, open(os.path.join(path_to_dataset, 'c1_onlyResized32.pckl'), 'wb'))
    #pickle.dump(c2, open(os.path.join(path_to_dataset, 'c2_onlyResized32.pckl'), 'wb'))
    #pickle.dump(c3, open(os.path.join(path_to_dataset, 'c3_onlyResized32.pckl'), 'wb'))
