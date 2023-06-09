# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pandas as pd
import torch
import time
from random_walker import RandomWalkerNode2Vec, RandomWalkerNode2VecPlus
from sampler import ARGEW, VanillaSampler
from node2vec import Node2Vec

from collections import Counter

# Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py


parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str)
parser.add_argument('--walker', type=str)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=80)
parser.add_argument('--context_size', type=int, default=10)
parser.add_argument('--walks_per_node', type=int, default=10)
parser.add_argument('--num_negative_samples', type=int, default=1)
parser.add_argument('--p', type=float, default=1)
parser.add_argument('--q', type=float, default=0.5)
parser.add_argument('--weights_rescale_low', type=int, default=1)
parser.add_argument('--weights_rescale_high', type=int, default=7)
parser.add_argument('--exp_base', type=int, default=2)

args = parser.parse_args()



# Prepare walk sample network data
edge_list = [(0,1, 3), (0,2, 3), (0,3, 3), (0,4, 3), (1,2, 3), (1,3, 3), (1,4, 3), (2,3, 3), (2,4, 3), (3,4, 3), (4,5, 2), (4,6, 2), (4,7, 2), (4,8, 2), (0,5, 1), (1,6, 1), (2,7, 1), (3,8, 1),
             (9,10, 3), (9,11, 3), (9,12, 3), (9,13, 3), (10,11, 3), (10,12, 3), (10,13, 3), (11,12, 3), (11,13, 3), (12,13, 3), (13,5, 2), (13,6, 2), (13,7, 2), (13,8, 2), (9,5, 1), (10,6, 1), (11,7, 1), (12,8, 1),
             (14,15, 3), (14,16, 3), (14,17, 3), (14,18, 3), (15,16, 3), (15,17, 3), (15,18, 3), (16,17, 3), (16,18, 3), (17,18, 3), (18,5, 2), (18,6, 2), (18,7, 2), (18,8, 2),  (14,5, 1), (15,6, 1), (16,7, 1), (17,8, 1),
             (5,6, 2), (5,7, 2), (5,8, 2), (6,7, 2), (6,8, 2), (7,8, 2)]
data = {'source':[s for s, t, w in edge_list], 'target':[t for s, t, w in edge_list], 'weight':[w for s, t, w in edge_list]}
df_edges = pd.DataFrame(data=data)

data = {'id':[], 'label':[]}
unique_nodes = set(df_edges['source'].tolist()).union(set(df_edges['target'].tolist()))
for node in unique_nodes:
    data['id'].append(node)
    data['label'].append(1)
data = {'id':list(unique_nodes), 'label':[1 for _ in range(0, len(unique_nodes))]}
df_nodes = pd.DataFrame(data=data)
print("df_nodes.shape: {}, df_edges.shape: {}".format(df_nodes.shape, df_edges.shape))
nodes_unique_list = list(df_nodes['id'].unique())
node_idx_dict = {x: i for i, x in enumerate(nodes_unique_list)}
df_nodes['node_idx'] = [node_idx_dict[row['id']] for i, row in df_nodes.iterrows()]
edge_node1_index = [node_idx_dict[row['source']] for i, row in df_edges.iterrows()]
edge_node2_index = [node_idx_dict[row['target']] for i, row in df_edges.iterrows()]
edge_index = torch.tensor([edge_node1_index, edge_node2_index])
edge_weight_list = [row['weight'] for i, row in df_edges.iterrows()]
edge_weights = torch.tensor(edge_weight_list)



# Define walker and sampler
if args.walker == 'node2vec':
    walker = RandomWalkerNode2Vec(args.p, args.q)
elif args.walker == 'node2vecplus':
    walker = RandomWalkerNode2VecPlus(args.p, args.q)
else:
    raise Exception('No walker!!!')

if args.sampler == 'argew':
    sampler = ARGEW(walker
                 , args.walks_per_node, args.num_negative_samples, args.walk_length, args.context_size
                 , edge_index, edge_weights
                 , args.weights_rescale_low, args.weights_rescale_high
                 , args.exp_base)
elif args.sampler == 'vanilla':
    sampler = VanillaSampler(walker
                            , args.walks_per_node, args.num_negative_samples, args.walk_length, args.context_size
                            , edge_index, edge_weights)
else:
    raise Exception('No sampler!!!')
model = Node2Vec(sampler = sampler,
                     edge_index = edge_index,
                     edge_weights = edge_weights,
                     embedding_dim = args.embedding_dim,
                     sparse = True)


# Sample walks 
batch = torch.tensor(df_nodes['node_idx'].tolist())
subsequences = sampler.pos_sample(batch)
if type(subsequences) == tuple: # sampler type is ARGEW, so pos_sample returns two variables
    _, subsequences = subsequences


# Aggregate
agg_dict = dict() # for each starting node, get the list of all nodes in its subsequences (include duplicates)
for i, row in enumerate(subsequences):
    main_node = row[0].item()
    
    if main_node in [0,1,2,3,4]:
        main_node_type = 'community#4bridge' if main_node == 4 else 'community#4internal'
    elif main_node in [9,10,11,12,13]:
        main_node_type = 'community#13bridge' if main_node == 13 else 'community#13internal'
    elif main_node in [14,15,16,17,18]:
        main_node_type = 'community#18bridge' if main_node == 18 else 'community#18internal'
    else:
        main_node_type = 'etc'

    if main_node_type not in agg_dict.keys():
        agg_dict[main_node_type] = []
    for coappear_node in row[1:]:
        coappear_node = coappear_node.item()
        if coappear_node in [0,1,2,3,4]:
            coappear_node_type = 'community#4bridge' if coappear_node == 4 else 'community#4internal'
        elif coappear_node in [9,10,11,12,13]:
            coappear_node_type = 'community#13bridge' if coappear_node == 13 else 'community#13internal'
        elif coappear_node in [14,15,16,17,18]:
            coappear_node_type = 'community#18bridge' if coappear_node == 18 else 'community#18internal'
        else:
            coappear_node_type = 'etc'
        agg_dict[main_node_type].append(coappear_node_type)


agg_counter_dict = {main_node_type: Counter(v) for main_node_type, v in agg_dict.items()}
for main_node_type, counter in agg_counter_dict.items():
    print('------- main_node_type = {} -------'.format(main_node_type))
    total = sum(counter.values())
    counter_prop = [(i, round(counter[i] / total, 3)) for i in counter]
    counter_prop = sorted(counter_prop, key=lambda tup:tup[0])
    print(counter_prop)




