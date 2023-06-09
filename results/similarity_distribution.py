# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json


parser = argparse.ArgumentParser()
parser.add_argument('--include_non_edge_pairs', type=int, default=1)
parser.add_argument('--bin_for_each_weight', type=int, default=0) #parser.add_argument('--weight_bin_size', type=int)
parser.add_argument('--weight_bin_cnt', type=int, default=8)
parser.add_argument('--df_edges_filepath', type=str)
parser.add_argument('--df_nodes_embedding_filepath', type=str)
args = parser.parse_args()

df_edges = pd.read_csv(args.df_edges_filepath, sep='\t')
# make source < target for every edge. when computing every combination, we follow this rule
df_edges['source_'] = [min([row['source'], row['target']]) for i, row in df_edges.iterrows()]
df_edges['target_'] = [max([row['source'], row['target']]) for i, row in df_edges.iterrows()]
df_edges = df_edges.drop(columns=['source', 'target'])
df_edges = df_edges.rename(columns={"source_": "source", "target_": "target"})
print('num of edges: {}'.format(df_edges.shape[0]))

df_nodes_embedding = pd.read_csv(args.df_nodes_embedding_filepath, sep='\t')
unique_nodes_sort = np.sort(df_nodes_embedding['id'].unique())
print('num of nodes: {}'.format(unique_nodes_sort.shape[0]))
dict_nodes_embedding = {row['id']: torch.tensor(json.loads(row['embedding'])) for i, row in df_nodes_embedding.iterrows()}

cos_sim_function = nn.CosineSimilarity(dim=0)
euclid_dist_function = nn.PairwiseDistance(p=2)

# for each node pair, compute embedding cos similarity and embedding euiclid distance
if args.include_non_edge_pairs != 0:
    print("include non edge pairs")
    node_pairs_data = {"source":[], "target":[], "cos_sim":[], "euclid_dist":[]}
    for i1 in range(0, unique_nodes_sort.shape[0]):
        for i2 in range(i1+1, unique_nodes_sort.shape[0]):
            source, target = unique_nodes_sort[i1], unique_nodes_sort[i2]
            node_pairs_data['source'].append(source)
            node_pairs_data['target'].append(target)
            source_embedding = dict_nodes_embedding[source] 
            target_embedding = dict_nodes_embedding[target]
            cos_sim = cos_sim_function(source_embedding, target_embedding).item()
            node_pairs_data['cos_sim'].append(cos_sim)
            euclid_dist = euclid_dist_function(source_embedding, target_embedding).item()
            node_pairs_data['euclid_dist'].append(euclid_dist)
    df_node_pairs = pd.DataFrame(data=node_pairs_data)
    df_node_pairs = pd.merge(df_node_pairs, df_edges, how='left', on=["source", "target"])
    df_node_pairs = df_node_pairs.fillna(0) # all node pairs with no edge
else:
    print("not include non edge pairs")
    node_pairs_data = {"source":[], "target":[], "weight":[], "cos_sim":[], "euclid_dist":[]}
    for i, row in df_edges.iterrows():
        source, target, weight = row['source'], row['target'], row['weight']
        node_pairs_data['source'].append(source)
        node_pairs_data['target'].append(target)
        node_pairs_data['weight'].append(weight)
        source_embedding = dict_nodes_embedding[source]
        target_embedding = dict_nodes_embedding[target]
        cos_sim = cos_sim_function(source_embedding, target_embedding).item()
        node_pairs_data['cos_sim'].append(cos_sim)
        euclid_dist = euclid_dist_function(source_embedding, target_embedding).item()
        node_pairs_data['euclid_dist'].append(euclid_dist)
    df_node_pairs = pd.DataFrame(data=node_pairs_data)

if args.bin_for_each_weight == 1: # define a bin for each unique weight
    bin_min_values = sorted(df_edges['weight'].unique())
else:
    splits = np.linspace(start=min(df_edges['weight']), stop=max(df_edges['weight']), num=args.weight_bin_cnt)
    bin_min_values = [min_val for i, min_val in enumerate(splits) if i < len(splits)-1]
if args.include_non_edge_pairs != 0:
    bin_min_values = [0] + bin_min_values
bin_data = {'bin_def_min_weight':[], 'min_weight':[], 'max_weight':[], 'node_pairs_cnt':[],
            'cos_sim_min':[], 'cos_sim_q1':[], 'cos_sim_median':[], 'cos_sim_mean':[], 'cos_sim_q3':[], 'cos_sim_max':[],
            'euc_dist_min':[], 'euc_dist_q1':[], 'euc_dist_median':[], 'euc_dist_mean':[], 'euc_dist_q3':[], 'euc_dist_max':[]
           }

for i in range(0, len(bin_min_values)):
    this_bin_min = bin_min_values[i]
    if this_bin_min == 0 or args.bin_for_each_weight == 1:
        df_this_bin_node_pairs = df_node_pairs.loc[df_node_pairs['weight']==this_bin_min].reset_index(drop=True)
    elif i < len(bin_min_values) - 1: # not last 
        next_bin_min = bin_min_values[i+1]
        df_this_bin_node_pairs = df_node_pairs.loc[(this_bin_min <= df_node_pairs['weight']) & (df_node_pairs['weight'] < next_bin_min)].reset_index(drop=True)
    else: # last
        df_this_bin_node_pairs = df_node_pairs.loc[this_bin_min <= df_node_pairs['weight']].reset_index(drop=True)
    this_bin_min_weight, this_bin_max_weight = np.min(df_this_bin_node_pairs['weight']), np.max(df_this_bin_node_pairs['weight'])
    this_bin_node_pairs_cnt = df_this_bin_node_pairs.shape[0]
    cos_sim_min, cos_sim_mean, cos_sim_max = np.min(df_this_bin_node_pairs['cos_sim']), np.mean(df_this_bin_node_pairs['cos_sim']), np.max(df_this_bin_node_pairs['cos_sim'])
    cos_sim_q1, cos_sim_median, cos_sim_q3 = np.percentile(df_this_bin_node_pairs['cos_sim'], [25, 50, 75])
    euc_dist_min, euc_dist_mean, euc_dist_max = np.min(df_this_bin_node_pairs['euclid_dist']), np.mean(df_this_bin_node_pairs['euclid_dist']), np.max(df_this_bin_node_pairs['euclid_dist'])
    euc_dist_q1, euc_dist_median, euc_dist_q3 = np.percentile(df_this_bin_node_pairs['euclid_dist'], [25, 50, 75])
    
    bin_data['bin_def_min_weight'].append(this_bin_min)
    bin_data['min_weight'].append(this_bin_min_weight)
    bin_data['max_weight'].append(this_bin_max_weight)
    bin_data['node_pairs_cnt'].append(this_bin_node_pairs_cnt)
    bin_data['cos_sim_min'].append(cos_sim_min)
    bin_data['cos_sim_q1'].append(cos_sim_q1)
    bin_data['cos_sim_median'].append(cos_sim_median)
    bin_data['cos_sim_mean'].append(cos_sim_mean)
    bin_data['cos_sim_q3'].append(cos_sim_q3)
    bin_data['cos_sim_max'].append(cos_sim_max)
    bin_data['euc_dist_min'].append(euc_dist_min)
    bin_data['euc_dist_q1'].append(euc_dist_q1)
    bin_data['euc_dist_median'].append(euc_dist_median)
    bin_data['euc_dist_mean'].append(euc_dist_mean)
    bin_data['euc_dist_q3'].append(euc_dist_q3)
    bin_data['euc_dist_max'].append(euc_dist_max)

df_bin = pd.DataFrame(data=bin_data)
print(df_bin)
savefilename = "sim_distribution_{}".format(args.df_nodes_embedding_filepath.split('/')[-1])
savefilename = '.'.join(savefilename.split('.')[:-1]) + '.csv'
df_bin.to_csv('/'.join(args.df_nodes_embedding_filepath.split('/')[:-1]) + '/{}'.format(savefilename), sep=',')





