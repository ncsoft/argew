# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--df_edges_filepath', type=str)
parser.add_argument('--df_nodes_filepath', type=str)
parser.add_argument('--node_features_matrix_pickle_filepath', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_repetitions', type=int, default=10)
args = parser.parse_args()


# reference: https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
class GCN(torch.nn.Module):
    def __init__(self, node_features_dim, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(node_features_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def gcn_train(model, optimizer, criterion, x, y, edge_index, edge_weight, train_indices):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(x.to(device), edge_index.to(device), edge_weight.to(device))  # Perform a single forward pass.
    loss = criterion(out[train_indices], y.to(device)[train_indices])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def gcn_predict(model, x, edge_index, edge_weight):
    model.eval()
    out = model(x.to(device), edge_index.to(device), edge_weight.to(device))
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    return pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: {}".format(device))

# For node classification, we perform 10 times (independently) of stratified sampling (50% train, 50% test).
# Reference: https://github.com/krishnanlab/node2vecplus_benchmarks/blob/main/script/eval_realworld_networks.py

df_nodes = pd.read_csv(args.df_nodes_filepath, sep='\t')
unique_nodes = sorted(list(df_nodes['id'].unique()))
unique_classes = df_nodes['label'].unique()
class_idx_dict = {cls:i for i, cls in enumerate(unique_classes)}
multilabel_matrix = np.array([[int(y in [class_idx_dict[cls] for cls in df_nodes.loc[df_nodes['id']==node_id].reset_index(drop=True)['label']]) for y in range(0, len(unique_classes))] for node_id in unique_nodes]) # shape: num nodes x num unique classes
node_idx_dict = {x: i for i, x in enumerate(unique_nodes)}

df_edges = pd.read_csv(args.df_edges_filepath, sep='\t')
edge_node1_index = [node_idx_dict[row['source']] for i, row in df_edges.iterrows()]
edge_node2_index = [node_idx_dict[row['target']] for i, row in df_edges.iterrows()]
edge_index = torch.tensor([edge_node1_index, edge_node2_index])
edge_weight_list = [row['weight'] for i, row in df_edges.iterrows()]
edge_weight = torch.tensor(edge_weight_list).to(torch.float32)


with open(args.node_features_matrix_pickle_filepath, 'rb') as f:
    node_features = pickle.load(f)
node_features = torch.from_numpy(node_features.astype(np.float32))

def train_and_eval(node_features, multilabel_matrix):
    f1_list = []
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    strat = StratifiedKFold(n_splits=2)
    for class_idx in range(0, multilabel_matrix.shape[1]): # for each unique class
        y = torch.from_numpy( multilabel_matrix[:,class_idx] )
        train_indices, eval_indices = next(strat.split(node_features, y))
        if len(set(y[train_indices])) == 1: # skip if training data contains only one class
            continue
        model = GCN(node_features_dim=node_features.shape[1], hidden_channels=700, num_classes=len(unique_classes))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, args.num_epochs+1):
            train_loss = gcn_train(model=model, optimizer=optimizer, criterion=criterion, 
                                   x=node_features, y=y, edge_index=edge_index, edge_weight=edge_weight, train_indices=train_indices)
            #print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}')

        eval_predictions = gcn_predict(model=model, x=node_features, edge_index=edge_index, edge_weight=edge_weight)
        eval_predictions = eval_predictions[eval_indices].to('cpu')
        eval_answers = y[eval_indices]
        tn, fp, fn, tp = confusion_matrix(y_true=eval_answers, y_pred=eval_predictions).ravel()
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        f1_list.append(f1_score(y_true=eval_answers, y_pred=eval_predictions))
    macro_f1 = np.mean(f1_list)
    micro_f1 = tp_sum / (tp_sum + (0.5 * (fp_sum + fn_sum)))
    return macro_f1, micro_f1

macro_f1_list, micro_f1_list = [], []
for _ in range(0, args.num_repetitions):
    # train and evaluate GCN model
    macro_f1, micro_f1 = train_and_eval(node_features, multilabel_matrix)
    macro_f1_list.append(macro_f1)
    micro_f1_list.append(micro_f1)
 
if 'amazon_photo_network' in args.df_nodes_filepath:
    network_name = 'amazon'
elif 'cora_network' in args.df_nodes_filepath:
    network_name = 'cora'
else:
    network_name = 'lineagew'
print("node classification for {} network".format(network_name))
print("nodes file: {}".format(args.df_nodes_filepath))
print("number of repetitions for stratified sampling: {}".format(args.num_repetitions))
print("average eval macro f1 score: {}".format(np.mean(macro_f1_list)))
print("average eval micro f1 score: {}".format(np.mean(micro_f1_list)))


