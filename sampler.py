# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import networkx as nx
import numpy as np

# Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py

class ARGEW:
    def __init__(self, walker, walks_per_node, num_negative_samples, walk_length, context_size
                , edge_index, edge_weights, weights_rescale_low, weights_rescale_high, exp_base):
        self.walker = walker
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.edge_index = edge_index
        self.edge_weights = edge_weights

        graph = nx.Graph()
        for i, edge in enumerate(edge_index.t()):
            id1 = edge[0].item()
            id2 = edge[1].item()
            weight = edge_weights[i].item()
            graph.add_edge(id1, id2, weight=weight)
        self.neighbor_info = {node: [[adj_item[0], adj_item[1]['weight']] for adj_item in list(graph.adj[node].items())] for node in list(graph.nodes)}

        # make a dictionary where each key is a unique edge weight value, and the value is the corresponding rescaled value
        def rescale(arr, low, high):
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) # rescale to [0, 1]
            arr = (arr * (high - low)) + low # rescale to [low, high]
            return arr
        weights_unique = np.array(edge_weights.unique())
        weights_rescaled = rescale(weights_unique, weights_rescale_low, weights_rescale_high)
        self.weights_rescaled_dict = {weights_unique[i]: weights_rescaled[i] for i in range(0, weights_unique.shape[0])}
        self.exp_base = exp_base

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        
        rw = self.walker.sample_random_walks(self.edge_index, self.edge_weights, batch, self.walk_length)
        
        weight_thres = np.percentile(self.edge_weights, 50) # if max weight found is <= threshold, we will not do augmentation
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size - 1 
        for j in range(num_walks_per_rw):
            sequences = rw[:, j:j + self.context_size]
            walks.append(sequences)
            aug_sequences = None
            for seq in sequences:
                seq = seq.tolist()
                for i in range(0, len(seq)):
                    is_first_node = (i == 0)
                    is_last_node = (i == len(seq) - 1)
                    curr_node = seq[i]
                    if not is_first_node:
                        prev_node = seq[i-1]
                    if not is_last_node:
                        next_node = seq[i+1]
                    # Among nodes in current node's neighbors that are also neighbors of the previous node and the current node, find the node with max weight with the current node
                    curr_node_neighbors_wgt = {neighbor: weight for neighbor, weight in self.neighbor_info[curr_node]}
                    if not is_first_node:
                        prev_node_neighbors = set([neighbor for neighbor, weight in self.neighbor_info[prev_node]])
                        curr_node_neighbors_wgt = {k: v for k, v in curr_node_neighbors_wgt.items() if k in prev_node_neighbors}
                    if not is_last_node:
                        next_node_neighbors = set([neighbor for neighbor, weight in self.neighbor_info[next_node]])
                        curr_node_neighbors_wgt = {k: v for k, v in curr_node_neighbors_wgt.items() if k in next_node_neighbors}
                    if len(curr_node_neighbors_wgt) == 0:
                        continue
                    # make a new seq: replace curr_node with max_weight_node
                    new_seq = [node for node in seq]
                    max_wgt = max(curr_node_neighbors_wgt.values())
                    if max_wgt <= weight_thres:
                        continue
                    new_seq[i] = max(curr_node_neighbors_wgt, key=curr_node_neighbors_wgt.get)
                    for _ in range(0, round(self.exp_base**self.weights_rescaled_dict[max_wgt])):
                        aug_sequences = torch.tensor([new_seq]) if aug_sequences is None else torch.cat([aug_sequences, torch.tensor([new_seq])])
                    aug_sequences = torch.cat([aug_sequences, torch.tensor([seq])])
            if aug_sequences is not None:
                walks.append(aug_sequences)

        return walks, torch.cat(walks, dim=0)
    
    def neg_sample(self, pos_walks):
        walks = []
        for sequences in pos_walks:
            for _ in range(0, self.num_negative_samples):
                random_tensor = torch.randint(high=self.adj.sparse_size(0), size=tuple(sequences.size()))
                # make it have same staring node as the corresponding positive sample
                random_tensor[:, 0] = sequences[:, 0]
                walks.append(random_tensor)
        return torch.cat(walks, dim=0)
    
    def sample(self, batch):
        pos_walks, pos_samples = self.pos_sample(batch)
        neg_samples = self.neg_sample(pos_walks)
        return pos_samples, neg_samples

class VanillaSampler:
    def __init__(self, walker, walks_per_node, num_negative_samples, walk_length, context_size, edge_index, edge_weights):
        self.walker = walker
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.edge_index = edge_index
        self.edge_weights = edge_weights

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        
        rw = self.walker.sample_random_walks(self.edge_index, self.edge_weights, batch, self.walk_length)
        
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size - 1 
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
    
    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        
        rw = torch.randint(self.adj.sparse_size(0),
                (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)
        
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size - 1 
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
    
    def sample(self, batch):
        return self.pos_sample(batch), self.neg_sample(batch)
    
