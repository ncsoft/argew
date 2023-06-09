# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import pickle

# from https://github.com/shchur/gnn-benchmark/blob/master/gnnbench/data/io.py
with np.load('./amazon_electronics_photo.npz') as loader:
    loader = dict(loader)
    adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

    if 'attr_data' in loader:
        # Attributes are stored as a sparse CSR matrix
        attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape'])
    elif 'attr_matrix' in loader:
        # Attributes are stored as a (dense) np.ndarray
        attr_matrix = loader['attr_matrix']
    else:
        attr_matrix = None

    if 'labels_data' in loader:
        # Labels are stored as a CSR matrix
        labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                               shape=loader['labels_shape'])
    elif 'labels' in loader:
        # Labels are stored as a numpy array
        labels = loader['labels']
    else:
        labels = None

    #node_names = loader.get('node_names')
    #attr_names = loader.get('attr_names')
    class_names = loader.get('class_names')
    #metadata = loader.get('metadata')

# make edge dataframe
df_adj_matrix = pd.DataFrame(adj_matrix.toarray())
G = nx.from_pandas_adjacency(df_adj_matrix)
df_edges = nx.to_pandas_edgelist(G)
df_edges.drop(columns=['weight'], inplace=True) # 'weight' is currently a column with all ones


# only include nodes that have at laest one edge
nodes_in_df_edges = set(df_edges['source'].tolist() + df_edges['target'].tolist())
row_indices_keep = []
for i in range(0, attr_matrix.shape[0]):
    if i in nodes_in_df_edges:
        row_indices_keep.append(i)

attr_matrix_np = attr_matrix.toarray()
attr_matrix_np = attr_matrix_np[row_indices_keep]
with open('amazon_node_features.pickle', 'wb') as f:
    pickle.dump(attr_matrix_np, f)





