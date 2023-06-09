# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import pandas as pd
import numpy as np

# Use onehot vector for each node (dimension = number of nodes)
df_nodes = pd.read_csv('nodes.tsv', sep='\t')
num_unique_nodes = len(df_nodes['id'].unique())
onehot_attribute_matrix = np.eye(num_unique_nodes) # diagonal matrix of ones, shape = number of nodes x number of nodes
with open('lineagew_node_features.pickle', 'wb') as f:
    pickle.dump(onehot_attribute_matrix, f)

