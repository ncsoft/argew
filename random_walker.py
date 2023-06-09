# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import random
import pandas as pd
import networkx as nx
from tqdm import tqdm

# node2vec with edge weights implementation reference: https://github.com/keras-team/keras-io/blob/master/examples/graph/node2vec_movielens.py

class RandomWalkerNode2Vec:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def next_step(self, graph, previous, current):
        neighbors = list(graph.neighbors(current))
    
        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(graph[current][neighbor]["weight"] / self.p)
            elif graph.has_edge(neighbor, previous):
                # The probability of visiting a local node.
                weights.append(graph[current][neighbor]["weight"])
            else:
                # Control the probability to move forward.
                weights.append(graph[current][neighbor]["weight"] / self.q)
    
        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def random_walk(self, graph, batch, walk_len):
        walks = []
        nodes = batch.tolist()
        for node in nodes: # tqdm(nodes):
            # Start the walk with the given node
            walk = [node]
            # Randomly walk for walk_len.
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = self.next_step(graph, previous, current)
                walk.append(next)
            # Add the walk to the generated sequence.
            walks.append(walk)
    
        return walks

    def sample_random_walks(self, edge_index, edge_weights, batch, walk_len):
        """
        INPUTS
        - edge_index: Graph connectivity in COO format with shape [2, num_edges]
        - edge_weights: edge weight values with shape [num_edges]
        - batch: starting nodes with shape [num starting nodes] including duplicates
        - walk_len: length of each walk
        - p: node2vec hyperparameter p
        - q: node2vec hyperparameter q
    
        OUTPUT
        - walks: sequence of walks with shape [num walks, walk length] 
        """
        # make nx graph object
        G = nx.Graph()
        for i, edge in enumerate(edge_index.t()):
            id1 = edge[0].item()
            id2 = edge[1].item()
            weight = edge_weights[i].item()
            G.add_edge(id1, id2, weight=weight)
    
        # run random walks
        walks = self.random_walk(G, batch, walk_len)
        return torch.tensor(walks)

class RandomWalkerNode2VecPlus:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def next_step(self, graph, previous, current):
        neighbors = list(graph.neighbors(current))
        if previous is not None:
            d_previous = graph.avg_weight_dict[previous]
        d_current = graph.avg_weight_dict[current]
        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            d_neighbor = graph.avg_weight_dict[neighbor]
    
            # check if (previous, neighbor) is tight
            previous_neighbor_tight = (graph.has_edge(neighbor, previous) and graph[previous][neighbor]['weight'] >= max(d_previous, d_neighbor))
            # check if (current, neighbor) is tight
            current_neighbor_tight = (graph[current][neighbor]['weight'] >= max(d_current, d_neighbor))
    
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(graph[current][neighbor]["weight"] / self.p)
            elif previous_neighbor_tight:
                weights.append(graph[current][neighbor]["weight"])
            elif not previous_neighbor_tight and not current_neighbor_tight:
                weights.append(graph[current][neighbor]["weight"] * min(1, 1/self.q))
            else:
                if not graph.has_edge(neighbor, previous):
                    weights.append(graph[current][neighbor]["weight"] / self.q)
                else:
                    weights.append(graph[current][neighbor]["weight"] * (1/self.q + ((1 - (1/self.q))*(graph[previous][neighbor]['weight'] / max(d_previous, d_neighbor)))))
    
        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def random_walk(self, graph, batch, walk_len):
        walks = []
        nodes = batch.tolist()
        for node in nodes: # tqdm(nodes):
            # Start the walk with the given node
            walk = [node]
            # Randomly walk for walk_len.
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = self.next_step(graph, previous, current)
                walk.append(next)
            # Add the walk to the generated sequence.
            walks.append(walk)
    
        return walks
         
    def sample_random_walks(self, edge_index, edge_weights, batch, walk_len):
        """
        INPUTS
        - edge_index: Graph connectivity in COO format with shape [2, num_edges]
        - edge_weights: edge weight values with shape [num_edges]
        - batch: starting nodes with shape [num starting nodes] including duplicates
        - walk_len: length of each walk
        - p: node2vec hyperparameter p
        - q: node2vec hyperparameter q
    
        OUTPUT
        - walks: sequence of walks with shape [num walks, walk length] 
        """
        # make nx graph object
        G = nx.Graph()
        for i, edge in enumerate(edge_index.t()):
            id1 = edge[0].item()
            id2 = edge[1].item()
            weight = edge_weights[i].item()
            G.add_edge(id1, id2, weight=weight)
        # For each node, compute the average weight of its edges. Store them in a dictionary.
        G.avg_weight_dict = {v: np.mean([G[v][neigh]['weight'] for neigh in G[v]]) for v in G.nodes()}
    
        # run random walks
        walks = self.random_walk(G, batch, walk_len)
        return torch.tensor(walks)


