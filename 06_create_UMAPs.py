# libraries

import os
import pickle
import umap

if __name__ == '__main__':

    # defining paths
    path_to_features = '/Users/saskia/unibe19/master_thesis/TKI_project/feature_maps/81_5c_featuresVectors_with_labels.pckl'
    path_to_UMAP_models = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/UMAP_models'
    path_to_UMAP_embeddings = '/Users/saskia/unibe19/master_thesis/TKI_project/UMAPs/UMAP_embeddings'

    # pickle features
    feature_vectors, labels, folder_labels = pickle.load(open(path_to_features, "rb"))

    # 2 components
    UMAP_trans_2c = umap.UMAP(n_neighbors=50, n_components=2, random_state=42, metric="euclidean").fit(feature_vectors)
    pickle.dump(UMAP_trans_2c, open(os.path.join(path_to_UMAP_models, '81_5c_UMAP_trans_2c.pckl'), 'wb'))
    pickle.dump(UMAP_trans_2c.embedding_, open(os.path.join(path_to_UMAP_embeddings, '81_5c_UMAP_2c_embedding.pckl'), 'wb'))

    # 3 components
    UMAP_trans_3c = umap.UMAP(n_neighbors=50, n_components=3, random_state=1711).fit(feature_vectors)
    pickle.dump(UMAP_trans_3c, open(os.path.join(path_to_UMAP_models, '81_5c_UMAP_trans_3c.pckl'), 'wb'))
    pickle.dump(UMAP_trans_3c.embedding_, open(os.path.join(path_to_UMAP_embeddings, '81_5c_UMAP_3c_embedding.pckl'), 'wb'))
