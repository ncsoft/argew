# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pandas as pd
import torch
import time
from random_walker import RandomWalkerNode2Vec, RandomWalkerNode2VecPlus
from sampler import ARGEW, VanillaSampler
from node2vec import Node2Vec


# Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py

LOG_INFO = ""

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
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--cuda_num', type=int, default=0)
parser.add_argument('--n_gpu', type=int, default=2)
parser.add_argument('--weights_rescale_low', type=int, default=1)
parser.add_argument('--weights_rescale_high', type=int, default=9)
parser.add_argument('--exp_base', type=int, default=2)
parser.add_argument('--df_nodes_filepath', type=str)
parser.add_argument('--df_edges_filepath', type=str)
parser.add_argument('--embeddings_save_dirpath', type=str)

args = parser.parse_args()


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model(pos_rw.to(device), neg_rw.to(device))
        # multigpu
        loss = loss.sum() / args.n_gpu
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_node_embedding():
    model.eval()
    batch = torch.tensor(list(df_nodes['node_idx'])).to(int).to(device)
    embeddings = model.module(batch).to(float).tolist()
    df_nodes['embedding'] = embeddings
    walker_nm = 'n2v' if args.walker == 'node2vec' else 'n2vplus'
    sampler_nm = 'AUG' if args.sampler == 'argew' else ''
    filename = "{}{}_embeddings_p{}_q{}_ed{}_wl{}_cs{}_wpn{}_nns{}_wrl{}_wsh{}_base{}_bs{}_lr{}_epochs{}.csv".format(walker_nm, sampler_nm, args.p, args.q, args.embedding_dim, args.walk_length, args.context_size, args.walks_per_node, args.num_negative_samples, args.weights_rescale_low, args.weights_rescale_high, args.exp_base, args.batch_size, args.lr, args.epochs)
    df_nodes.to_csv("{}{}".format(args.embeddings_save_dirpath, filename), sep='\t')


if __name__ == "__main__":
    df_nodes = pd.read_csv(args.df_nodes_filepath, sep='\t') # each row is each <node, label> pair
    df_edges = pd.read_csv(args.df_edges_filepath, sep='\t') # each row is each edge
    
    nodes_unique_list = list(df_nodes['id'].unique())
    node_idx_dict = {x: i for i, x in enumerate(nodes_unique_list)}
    df_nodes['node_idx'] = [node_idx_dict[row['id']] for i, row in df_nodes.iterrows()]
    
    edge_node1_index = [node_idx_dict[row['source']] for i, row in df_edges.iterrows()]
    edge_node2_index = [node_idx_dict[row['target']] for i, row in df_edges.iterrows()]
    edge_index = torch.tensor([edge_node1_index, edge_node2_index])

    edge_weight_list = [row['weight'] for i, row in df_edges.iterrows()]
    edge_weights = torch.tensor(edge_weight_list)

    device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'
    print("device: {}".format(device))
    LOG_INFO += "device: {}".format(device) + '\n'

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
    print("model: {}".format(model))
    LOG_INFO += "model: {}".format(model) + '\n'
    device_ids = [x for x in range(args.cuda_num, args.cuda_num + args.n_gpu)]
    print("GPU device_ids: {}".format(device_ids))
    LOG_INFO += "GPU device_ids: {}".format(device_ids) + '\n'
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    
    print("walker={}, sampler={}, embedding_dim={}, walk_length={}, context_size={}, walks_per_node={}, num_negative_samples={}, p={}, q={}, weights_rescale_low={}, weights_rescale_high={}, exp_base={}".format(args.walker, args.sampler, args.embedding_dim, args.walk_length, args.context_size, args.walks_per_node, args.num_negative_samples, args.p, args.q, args.weights_rescale_low, args.weights_rescale_high, args.exp_base))
    LOG_INFO += "walker={}, sampler={}, embedding_dim={}, walk_length={}, context_size={}, walks_per_node={}, num_negative_samples={}, p={}, q={}, weights_rescale_low={}, weights_rescale_high={}, exp_base={}".format(args.walker, args.sampler, args.embedding_dim, args.walk_length, args.context_size, args.walks_per_node, args.num_negative_samples, args.p, args.q, args.weights_rescale_low, args.weights_rescale_high, args.exp_base) + '\n'
    loader = model.module.loader(batch_size=args.batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    print("batch_size={}, lr={}, epochs={}".format(args.batch_size, args.lr, args.epochs))
    LOG_INFO += "batch_size={}, lr={}, epochs={}".format(args.batch_size, args.lr, args.epochs) + '\n'

    print("------ Start training ------")
    LOG_INFO += "------ Start training ------" + '\n'
    for epoch in range(0, args.epochs):
        start = time.time()
        loss = train()
        minutes = (time.time() - start) / 60
        print(f'After Epoch: {epoch:02d}, Loss: {loss:.4f} ... This epoch took {minutes:.2f} minutes')
        LOG_INFO += f'After Epoch: {epoch:02d}, Loss: {loss:.4f} ... This epoch took {minutes:.2f} minutes' + '\n'
        if epoch > 0 and prev_loss <= loss:
            break
        prev_loss = loss
    
    print("------ Finished training ------")
    LOG_INFO += "------ Finished training ------" + '\n'
    save_node_embedding()

    walker_nm = 'n2v' if args.walker == 'node2vec' else 'n2vplus'
    sampler_nm = 'AUG' if args.sampler == 'argew' else ''
    logfilename = "trainlogs_{}{}_p{}_q{}_ed{}_wl{}_cs{}_wpn{}_nns{}_wrl{}_wsh{}_base{}_bs{}_lr{}_epochs{}.txt".format(walker_nm, sampler_nm, args.p, args.q, args.embedding_dim, args.walk_length, args.context_size, args.walks_per_node, args.num_negative_samples, args.weights_rescale_low, args.weights_rescale_high, args.exp_base, args.batch_size, args.lr, args.epochs)
    with open("{}trainlogs/{}".format(args.embeddings_save_dirpath, logfilename), 'w') as fw:
        fw.write(LOG_INFO)


