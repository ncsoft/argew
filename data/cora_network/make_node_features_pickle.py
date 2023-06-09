# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch_geometric.datasets import Planetoid
import pickle

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
features = data['x'].numpy()
with open('cora_node_features.pickle', 'wb') as f:
    pickle.dump(features, f)

