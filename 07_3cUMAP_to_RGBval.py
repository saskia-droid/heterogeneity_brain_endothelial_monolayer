# libraries

import os
import pickle
import numpy as np

if __name__ == '__main__':

    # defining paths
    path_to_3cUMAPmodel = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/UMAP_models/81_5c_UMAP_trans_3c.pckl'
    path_to_UMAP_embeddings = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/UMAP_embeddings/'

    UMAP_trans_3c = pickle.load(open(path_to_3cUMAPmodel, 'rb'))

    RGB_UMAP_embed = UMAP_trans_3c.embedding_.copy()
    for i in range(3):
        min_val = np.min(RGB_UMAP_embed[:, i])
        max_val = np.max(RGB_UMAP_embed[:, i])
        divid_val = max_val - min_val
        RGB_UMAP_embed[:, i] = RGB_UMAP_embed[:, i] - min_val
        RGB_UMAP_embed[:, i] = RGB_UMAP_embed[:, i] / divid_val

    pickle.dump(RGB_UMAP_embed, open(os.path.join(path_to_UMAP_embeddings, '81_5c_UMAP_3c_RGB_embedding.pckl'), 'wb'))