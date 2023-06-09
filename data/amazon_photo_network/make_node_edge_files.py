# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx

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

# make node datafarme
nodes_in_df_edges = set(df_edges['source'].tolist() + df_edges['target'].tolist())
nodes_data = {'id': [], 'label': []}
for i in range(0, attr_matrix.shape[0]):
    # only include nodes that have at laest one edge
    if i not in nodes_in_df_edges:
        continue
    product_category = class_names[labels[i]]
    nodes_data['id'].append(i)
    nodes_data['label'].append(product_category)
    
df_nodes = pd.DataFrame(data=nodes_data)

# compute edge weights: cosine similarity of the two bag-of-words feature vectors
def cosine_similarity(list1, list2):
    return np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))

cos_sim_list = []
for i, row in df_edges.iterrows():
    source, target = row['source'], row['target']
    source_bow_feature = attr_matrix[source,:].toarray()[0].tolist()
    target_bow_feature = attr_matrix[target,:].toarray()[0].tolist()
    cos_sim = cosine_similarity(source_bow_feature, target_bow_feature)
    cos_sim_list.append(cos_sim)

df_edges['weight'] = cos_sim_list

# Out of the 119082 edges, two of them have weight (cosine similarity) = 0. We decided to remove those two edges.
df_edges = df_edges.loc[df_edges['weight'] > 0].reset_index(drop=True)
df_nodes = df_nodes.merge(pd.DataFrame(data={'id': list(set(df_edges['source'].tolist() + df_edges['target'].tolist()))}), on=['id'])

df_nodes.to_csv('nodes.tsv', sep='\t', index=False)
df_edges.to_csv('edges.tsv', sep='\t', index=False)





