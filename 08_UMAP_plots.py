# libraries

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':

    # defining paths
    path_to_UMAPs = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/'
    path_to_UMAP_models = os.path.join(path_to_UMAPs, 'UMAP_models')
    path_to_UMAP_3c_RGB_embed = os.path.join(path_to_UMAPs,'UMAP_embeddings/81_5c_UMAP_3c_RGB_embedding.pckl')
    path_to_UMAP_plots = os.path.join(path_to_UMAPs,'plots')
    path_to_features = '/Users/saskia/unibe19/master_thesis/TKI_project/feature_maps/81_5c_featuresVectors_with_labels.pckl'

    class_names = ['NS', 'IL-1b High', 'IL-1b Low', 'TNF', 'TNF+IFN']

    UMAP_trans_2c = pickle.load(open(os.path.join(path_to_UMAP_models, '81_5c_UMAP_trans_2c.pckl'), 'rb'))
    UMAP_3c_RGB_embed = pickle.load(open(path_to_UMAP_3c_RGB_embed, 'rb'))
    feature_vectors, labels, folder_labels = pickle.load(open(path_to_features, "rb"))

    # scatter plots for the different classes
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    for i in [0, 1, 2, 3, 4]:
        mask = labels == i
        selected_umap = UMAP_trans_2c.embedding_[mask, :]
        rest_umap = UMAP_trans_2c.embedding_[~mask, :]
        ax[0][i].scatter(rest_umap[:, 0], rest_umap[:, 1], c='lightgray', s=2, alpha=0.1)
        ax[0][i].scatter(selected_umap[:, 0], selected_umap[:, 1], c=np.mean(UMAP_3c_RGB_embed[mask, :], axis=0), s=10, alpha=0.5)
        ax[0][i].set_title(class_names[i])
        sns.kdeplot(x=UMAP_trans_2c.embedding_[mask, 0], y=UMAP_trans_2c.embedding_[mask, 1], fill=True,
                    ax=ax[1][i]).set(
            title=class_names[i])

        ax[0][i].set_xlim([-8, 10])
        ax[1][i].set_xlim([-8, 10])
        ax[0][i].set_ylim([-10, 15])
        ax[1][i].set_ylim([-10, 15])

    plt.savefig(os.path.join(path_to_UMAP_plots, '81_5c_UMAP_plots_3classes.png'))

    # scatter plots for the different classes
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    for i in [0, 1, 2, 3, 4]:
        mask = labels == i
        selected_umap = UMAP_trans_2c.embedding_[mask, :]
        rest_umap = UMAP_trans_2c.embedding_[~mask, :]
        color = UMAP_3c_RGB_embed[mask, :]
        ax[i].scatter(rest_umap[:, 0], rest_umap[:, 1], c='lightgray', s=2, alpha=0.1)
        ax[i].scatter(selected_umap[:, 0], selected_umap[:, 1], c=color, s=10, alpha=0.5)
        ax[i].set_title(class_names[i])
        #sns.kdeplot(x=UMAP_trans_2c.embedding_[mask, 0], y=UMAP_trans_2c.embedding_[mask, 1], fill=True, ax=ax[1][i]).set(title=class_names[i])

        ax[i].set_xlim([-8, 10])
        #ax[1][i].set_xlim([-8, 12])
        ax[i].set_ylim([-10, 15])
        #ax[1][i].set_ylim([-4, 16])

    plt.savefig(os.path.join(path_to_UMAP_plots, '81_5c_UMAP_plots_3classes_UMAP3c_colored.png'))
