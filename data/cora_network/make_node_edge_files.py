# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch_geometric.datasets import Planetoid
import pandas as pd

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

edge_index = data['edge_index'].T.numpy()
df_edges = pd.DataFrame(edge_index, columns=['source', 'target'])

# The Cora network data's edge_index provided in the torch_geometric library has duplicate rows for each edge.
# That is: undirected edge in both directions.
# This can be verified via the following commented lines: 
# df_edges['unordered_pair'] = ['{}_{}'.format(min(row['source'], row['target']), max(row['source'], row['target'])) for i, row in df_edges.iterrows()]
# df_edges.groupby(['unordered_pair']).size().reset_index(name='counts').groupby(['counts']).size().reset_index(name='num_pairs')

# So we remove the duplicates and keep one row for each node pair that has an edge.
# Make the smaller node id be the 'source', and the bigger node id be the 'target'.
df_edges['small_id'] = [min(row['source'], row['target']) for i, row in df_edges.iterrows()]
df_edges['big_id'] = [max(row['source'], row['target']) for i, row in df_edges.iterrows()]
df_edges.drop(columns=['source', 'target'], inplace=True)
df_edges.rename(columns={'small_id': 'source', 'big_id': 'target'}, inplace=True)
df_edges.drop_duplicates(inplace=True)

# compute edge weights: for the two nodes' feature vector, how many values (out of the 1433 dimensions) are the same?
def common_dim_cnt(arr1, arr2):
    return len([i for i in range(0, len(arr1)) if arr1[i] == arr2[i]])

features = data['x'].numpy() 
weights = []
for i, row in df_edges.iterrows():
    source_feature = features[row['source']]
    target_feature = features[row['target']]
    weight = common_dim_cnt(source_feature, target_feature)
    weights.append(weight)
df_edges['weight'] = weights

node_labels = data['y'].numpy()
df_nodes = pd.DataFrame(data={'id': list(range(0, node_labels.shape[0])), 'label': node_labels.tolist()})

df_nodes.to_csv('nodes.tsv', sep='\t', index=False)
df_edges.to_csv('edges.tsv', sep='\t', index=False)

