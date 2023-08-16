# ARGEW: Augmentation of Random walks by Graph Edge Weights
This repository contains scripts for the results presented in the paper _Node Embedding for Homophilous Graphs with ARGEW: Augmentation of Random walks by Graph Edge Weights_. (arXiv: https://arxiv.org/abs/2308.05957)


## Python libraries
Here are the python libraries that should be installed for the scripts in this repository.
- _torch_ (https://pytorch.org/get-started/locally/)
- _torch_geometric_ (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- _numpy_ (https://numpy.org/install/)
- _pandas_ (https://pandas.pydata.org/docs/getting_started/install.html)
- _networkx_ (https://networkx.org/documentation/stable/install.html)
- _tqdm_ (https://github.com/tqdm/tqdm#installation)
- _sklearn_ (https://scikit-learn.org/stable/install.html)


## Datasets
In the paper, we use three networks with high homophily for our experiments.
1. LineageW network
    - This network is based on the user log data of a Massively Multiplayer Online Role-Playing Game (MMORPG) called _LineageW_ (https://lineagew.plaync.com/).
    - Each node is a game account, and edges represent in-game interactions between the two.
2. Cora network
    - Each node is a publication, and edges represent citations. (Reference: http://eliassi.org/papers/ai-mag-tr08.pdf)  
    - We use the Planetoid version provided in the PyTorch Geometric package (https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid)
3. Amazon Photo network
    - Each node is a photo-related product in Amazon, and edges mean the two products are often bought together. (Reference: https://arxiv.org/pdf/1811.05868.pdf)
    - We use the `amazon_electronics_photo.npz` file provided in https://github.com/shchur/gnn-benchmark/tree/master/data/npz.

For training node2vec (either with or without ARGEW), we need to prepare two tsv files that represent the data for each network. 
- `nodes.tsv`: This is a tsv file about the node information with columns 'id' and 'label'. Each row is a unique <node id, label> pair. If a node has multiple labels (i.e. multi-label classification), then a node will appear in multiple rows in the file. 
    - Note: Training node2vec does not require node labels. But in our implementation, we included the label column in `nodes.tsv` to conveniently perform the node classification downstream task.
- `edges.tsv`: This is a tsv file about the edge information with columns 'source', 'target', and 'weight'. Each row is a unique edge. The 'source' and 'target' columns are node id values present in the 'id' column of the `nodes.tsv` file.

For training supervised GCN, we additionally need a pickle file that contains the node attribute matrix (row count: number of nodes, column count: attribute dimension).

For the LineageW network, we have uploaded the `nodes.tsv` and `edges.tsv` files (anonymized) in the `data/lineagew_network/` folder. For the other two networks, we have uploaded the script `make_node_edge_files.py` that creates the appropriate `nodes.tsv` and `edges.tsv`. (See the `data/cora_network/` and `data/amazon_photo_network/` folders respectively. Note that for the Amazon Photo network, the script assumes the `amazon_electronics_photo.npz` file in the same directory.) Also, for each dataset folder, we uploaded `make_node_features_pickle.py` that creates the pickle file of the node attribute matrix (as a numpy 2d array).




## Training and saving node embeddings
Once the node and edge tsv files are ready, we can train and save the node embeddings via running the `train_node2vec.py` python script. We run the script by providing the following arguments which will be processed by argparse.
- --sampler (STR): choose whether to perform our proposed augmentation method on the completed random walks - "vanilla" (not perform) or "argew" (do perform)
- --walker (STR): choose random walk strategy - "node2vec" or "node2vecplus"
- --embedding_dim (INT, default=128): dimension of the node embedding vectors
- --walk_length (INT, default=80): length of each walk
- --context_size (INT, default=10): context window length
- --walks_per_node (INT, default=10): number of walks per node (as a starting point)
- --num_negative_samples (INT, default=1): number of negative examples per positive (for the negative sampling trick)
- --p (FLOAT, default=1): return hyperparameter _p_
- --q (FLOAT, default=0.5): in-out hyperparameter _q_
- --batch_size (INT, default=128): batch size 
- --lr (FLOAT, default=0.01): learning rate
- --epochs (INT, default=10): number of epochs
- --cuda_num (INT, default=0): GPU id
- --n_gpu (INT, default=2): number of GPU devices to use
- --weights_rescale_low (INT, default=1): lower bound of weight rescale range for ARGEW 
- --weights_rescale_high (INT, default=7): upper bound of weight rescale range for ARGEW 
- --exp_base (INT, default=2): exponent base of augmentation count for ARGEW 
- --df_nodes_filepath (STR): path of the `nodes.tsv` file
- --df_edges_filepath (STR): path of the `edges.tsv` file
- --embeddings_save_dirpath (STR): path of the directory where the node embedding file will be saved

For example, to train node2vec with ARGEW, we can run: 

`python3 train_node2vec.py --sampler argew --walker node2vec --walks_per_node 1 --batch_size 8 --df_nodes_filepath /path/to/nodes.tsv --df_edges_filepath /path/to/edges.tsv --embeddings_save_dirpath /path/to/save/directory/`

This will save the embedding tsv file in the --embeddings_save_dirpath folder. The embedding tsv file is basically the `nodes.tsv` file with the corresponding node's embedding vector added as a column.


## Checking distribution of embedding similarities/distances 

The ARGEW method's goal is to support a random walk based node embedding method in such a way that nodes with stronger edges end up with closer embeddings. To verify whether ARGEW achieves this, we check the distribution of the similarities/distances of the embedding pairs. We equally split edge weights into bins (so each bin represents a particular range of edge weights), and we compute the distribution of the cosine similarities (of two embeddings) for node pairs that have edge weight within the corresponding bin. 

We can do so by running the `results/similarity_distribution.py` script, which creates the weight bins, and calculates the statistics (minimum, Q1, median, mean, Q3, maximum) of the cosine similarities and Euclidean distances for each bin. This script takes the following arguments:
- --include_non_edge_pairs (INT, default=1): choose whether to calculate statistics for node pairs that don't have an edge (i.e. weight = 0) - 1 (yes) or 0 (no)
- --bin_for_each_weight (INT, default=0): choose whether to make one bin per unique edge weight - 1 (yes) or 0 (no)
- --weight_bin_cnt (INT, default=8): number of bins to create
- --df_edges_filepath (STR): path of the `edges.tsv` file
- --df_nodes_embedding_filepath (STR): path of the node embedding file


## Node classification 

A one vs rest logistic regression classifier with L2 regularization is trained where the node embedding vector is used as (the only) features. We perform multiple times of stratified sampling that splits the data to 50\% training and 50\% testing. The micro-averaged F1 score and the macro-averaged F1 score are printed. This is done by running the `results/node_classification.py` script, which takes the following arguments:
- --df_nodes_embedding_filepath (STR): path of the node embedding file 
- --num_repetitions (INT, default=10): number of stratified sampling 


### GCN classification model

For further comparison, we also train a GCN classification model. This is done by running the `gcn_classification/gcn_classification.py` script, which trains the model and prints the average micro and macro F1 scores. The script takes the following arguments:
- --df_edges_filepath (STR): path of the `edges.tsv` file
- --df_nodes_filepath (STR): path of the `nodes.tsv` file
- --node_features_matrix_pickle_filepath (STR): path of the pickle file of the node attribute matrix  
- --num_epochs (INT, default=100): number of epochs for training
- --num_repetitions (INT, default=10): number of stratified sampling 


## Walk experiments

Lastly, to see why ARGEW works consistently well, we perform walk experiments on a synthetic network with clear structural roles and explore the coappearance distributions for diverse hyperparameter settings. This is done by running the `walk_experiment.py` script, which takes the following arguments:
- --sampler (STR): choose whether to perform our proposed augmentation method on the completed random walks - "vanilla" (not perform) or "argew" (do perform)
- --walker (STR): choose random walk strategy - "node2vec" or "node2vecplus"
- --embedding_dim (INT, default=128): dimension of the node embedding vectors
- --walk_length (INT, default=80): length of each walk
- --context_size (INT, default=10): context window length
- --walks_per_node (INT, default=10): number of walks per node (as a starting point)
- --num_negative_samples (INT, default=1): number of negative examples per positive (for the negative sampling trick)
- --p (FLOAT, default=1): return hyperparameter _p_
- --q (FLOAT, default=0.5): in-out hyperparameter _q_
- --weights_rescale_low (INT, default=1): lower bound of weight rescale range for ARGEW 
- --weights_rescale_high (INT, default=7): upper bound of weight rescale range for ARGEW 
- --exp_base (INT, default=2): exponent base of augmentation count for ARGEW 






