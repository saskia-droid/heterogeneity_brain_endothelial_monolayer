
# libraries

import os
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np

def min_img_size(datadir):
    w, h = (2000, 2000)
    for root, dirs, files in os.walk(datadir):
        for name in files:
            if '.png' in name:
                filename = os.path.join(root, name)
                img = Image.open(filename)
                new_w, new_h = img.size
                if new_w < w:
                    w = new_w
                if new_h < h:
                    h = new_h
    img_size = (h, w)
    return img_size

if __name__ == '__main__':

    # defining paths
    path_to_Xs_and_Ys = '/Users/saskia/unibe19/master_thesis/TKI_project/Xs_and_Ys/'
    path_to_features = '/Users/saskia/unibe19/master_thesis/TKI_project/feature_maps/'

    # upload the model
    models_path = '/Users/saskia/unibe19/master_thesis/TKI_project/UBELIX/models/'
    model = tf.keras.models.load_model(os.path.join(models_path, '78_3c_l2Normalize_lrelu_bestEpoch.h5'))

    # extract trained layers that we need
    model.layers.pop(0)
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.Input(shape=(None, None, 1)))  # change input dim
    for layer in model.layers[1:-2]:  # -2 -> third last layer as output
        new_model.add(layer)

    # upload the images
    imgdir = "/Users/saskia/unibe19/master_thesis/TKI_project/data/dataset_PC_HN"
    img_size = min_img_size(imgdir)
    img_set = tf.keras.utils.image_dataset_from_directory(imgdir, color_mode='grayscale',
                                                          batch_size=1, image_size=img_size,
                                                          crop_to_aspect_ratio=True)

    info = pickle.load(open(os.path.join(imgdir, "info.pckl"), "rb"))
    class_names = ['NS', 'IL-1b High', 'IL-1b Low', 'TNF', 'TNF+IFN']
    classes_to_keep = []
    for folder, detail in info.items():
        if detail['stim'] in class_names:
            classes_to_keep.append(folder)

    # normalization
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    set_n = img_set.map(lambda im, lbl: (normalization_layer(im), lbl))

    # retrieve labels and remove data which are not part of the 3 classes of interest
    Xs = []
    Ys = []
    for x, y in set_n:
        if y in classes_to_keep:
            Xs.append(x.numpy())
            Ys.append(y.numpy())
    Xs = np.concatenate(Xs, axis=0)
    Ys = np.concatenate(Ys, axis=0)

    # pickling images (Xs) and labels (Ys)
    pickle.dump(Xs, open(os.path.join(path_to_Xs_and_Ys, '81_5c_images.pckl'), 'wb'))
    pickle.dump(Ys, open(os.path.join(path_to_Xs_and_Ys, '81_5c_labels.pckl'), 'wb'))

    # get feature map
    feature_maps = new_model.predict(Xs)
    feature_vectors = feature_maps.reshape(-1, feature_maps.shape[-1])

    # adjust labels
    labels = [info[i]['stim_class'] for i in Ys]
    labels = np.repeat(labels, feature_maps.shape[1] * feature_maps.shape[2])
    folder_labels = np.repeat(Ys, feature_maps.shape[1] * feature_maps.shape[2])

    features_with_labels = [feature_vectors, labels, folder_labels]

    pickle.dump(features_with_labels, open(os.path.join(path_to_features, '81_5c_featuresVectors_with_labels.pckl'), 'wb'))

