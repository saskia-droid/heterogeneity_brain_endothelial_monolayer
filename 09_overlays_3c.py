# libraries

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

if __name__ == '__main__':

    # defining paths
    path_to_UMAPs = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/'
    path_to_UMAP_3c_RGB_embed = os.path.join(path_to_UMAPs, 'UMAP_embeddings/81_5c_UMAP_3c_RGB_embedding.pckl')
    path_to_Xs_and_Ys = '/Users/saskia/unibe19/master_thesis/TKI_project/Xs_and_Ys/'
    path_to_features = '/Users/saskia/unibe19/master_thesis/TKI_project/feature_maps/81_5c_featuresVectors_with_labels.pckl'
    path_to_overlays_3c = os.path.join(path_to_UMAPs, 'overlays_3c')
    path_to_info = "/Users/saskia/unibe19/master_thesis/TKI_project/data/dataset_PC_HN/info.pckl"

    info = pickle.load(open(path_to_info, "rb"))
    class_names = ['NS', 'IL-1b High', 'IL-1b Low', 'TNF', 'TNF+IFN']

    # pickling
    UMAP_3c_RGB_embed = pickle.load(open(path_to_UMAP_3c_RGB_embed, 'rb'))
    Xs = pickle.load(open(os.path.join(path_to_Xs_and_Ys, '81_5c_images.pckl'), 'rb'))
    Ys = pickle.load(open(os.path.join(path_to_Xs_and_Ys, '81_5c_labels.pckl'), 'rb'))
    [feature_vectors, labels, folder_labels] = pickle.load(open(path_to_features, 'rb'))

    # add channels to the grayscale images:
    imgs_RGB = []
    for i in range(len(Xs)):
        imgs_RGB.append(cv2.cvtColor(Xs[i], cv2.COLOR_GRAY2RGB))

    # reshape UMAP:
    constante = 32 # have to retrieve why it is 32 again
    reshaped_umap = []
    h = math.ceil(Xs.shape[1]/constante)
    w = math.ceil(Xs.shape[2]/constante)
    for i in range(len(Xs)):
        lower_index = i * h * w
        upper_index = (i+1) * h * w
        reshaped_umap.append(np.reshape(UMAP_3c_RGB_embed[lower_index:upper_index, :], (h, w, 3)))

    resized_umap_imgs = []
    for i in range(len(reshaped_umap)):
        resized_umap_imgs.append(cv2.resize(reshaped_umap[i], (1388, 1040)))

    alpha = 0.3
    beta = 1 - alpha
    alpha = alpha * 10  # to overcome a saturated original image

    for i in range(len(imgs_RGB)):
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        axs[0].axis('off')
        axs[0].imshow(imgs_RGB[i])

        axs[1].axis('off')
        axs[1].imshow(reshaped_umap[i])

        axs[2].axis('off')

        img_fused = (alpha * imgs_RGB[i] + beta * resized_umap_imgs[i]) / 2

        axs[2].imshow(img_fused)

        plt.suptitle('folder: ' + str(Ys[i]) + ', class: ' + info[Ys[i]]['stim'])

        plt.savefig(os.path.join(path_to_overlays_3c, f'81_5c_overlay_3c_{i}.png'))

