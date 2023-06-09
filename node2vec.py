# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes

# Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py
# We modify the pyg pacakge's node2vec implementation so that edge weights are considered in the unnormalized probabilities.
# Also, we modify the forward function so that pytorch DataParallel can be used (Reference: https://github.com/ankur6ue/node2vec_dataparallel)

EPS = 1e-15

class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        sampler (ARGEW instance or VanillaSampler instance): Which sampling method to use
        edge_index (LongTensor): The edge indices.
        edge_weights (DoubleTensor): The edge weights 
        embedding_dim (int): The size of each embedding vector.
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(self, sampler, edge_index, edge_weights, embedding_dim, 
                 num_nodes=None, sparse=False):
        super().__init__()

        self.sampler = sampler
        self.edge_index = edge_index
        self.edge_weights = edge_weights

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')
        self.sampler.adj = self.adj

        self.embedding_dim = embedding_dim

        self.embedding = Embedding(N, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        
    def forward(self, data1=None, data2=None):
        """
        If called with data1 and data2, then compute the loss where data1 is the positive examples and data2 is the negative examples.
        If called with only data1, then just return the embeddings of the nodes whose indices are the values in data1
        Otherwise, just return the embeddings for all nodes.
        """
        emb = self.embedding.weight
        if data1 is None and data2 is None:
            return emb
        elif data2 is None:
            return emb.index_select(0, data1)
        loss = self.loss(pos_rw=data1, neg_rw=data2)
        return loss
    
    def loader(self, **kwargs):
        return DataLoader(range(self.adj.sparse_size(0)),
                collate_fn=self.sample, **kwargs)
    
    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.sampler.sample(batch)
    
    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""
        
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        
        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                self.embedding_dim)
        
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        
        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        
        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                self.embedding_dim)
        
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        
        return pos_loss + neg_loss
    
    
    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
            multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression
        
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                **kwargs).fit(train_z.detach().cpu().numpy(),
                        train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                test_y.detach().cpu().numpy())
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')


