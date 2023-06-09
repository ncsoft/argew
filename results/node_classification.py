# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import json


parser = argparse.ArgumentParser()
parser.add_argument('--df_nodes_embedding_filepath', type=str)
parser.add_argument('--num_repetitions', type=int, default=10)
args = parser.parse_args()

# For node classification, we perform 10 times (independently) of stratified sampling (50% train, 50% test).
# Reference: https://github.com/krishnanlab/node2vecplus_benchmarks/blob/main/script/eval_realworld_networks.py

df_nodes_embedding = pd.read_csv(args.df_nodes_embedding_filepath, sep='\t')
unique_nodes = df_nodes_embedding['id'].unique()
X_embedding = np.array([json.loads(df_nodes_embedding.loc[df_nodes_embedding['id']==node_id].reset_index(drop=True)['embedding'][0]) for node_id in unique_nodes])
unique_classes = df_nodes_embedding['label'].unique()
class_idx_dict = {cls:i for i, cls in enumerate(unique_classes)}
multilabel_matrix = np.array([[int(y in [class_idx_dict[cls] for cls in df_nodes_embedding.loc[df_nodes_embedding['id']==node_id].reset_index(drop=True)['label']]) for y in range(0, len(unique_classes))] for node_id in unique_nodes]) # shape: num nodes x num unique classes

def train_and_eval(X_embedding, multilabel_matrix):
    f1_list = []
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    strat = StratifiedKFold(n_splits=2)
    for class_idx in range(0, multilabel_matrix.shape[1]): # for each unique class
        y = multilabel_matrix[:,class_idx]
        train_indices, eval_indices = next(strat.split(X_embedding, y))
        if len(set(y[train_indices])) == 1: # skip if training data contains only one class
            continue
        lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=500)
        lr.fit(X_embedding[train_indices], y[train_indices])
        eval_predictions = lr.predict(X_embedding[eval_indices])
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
    # train and evaluate logistic regression model
    macro_f1, micro_f1 = train_and_eval(X_embedding, multilabel_matrix)
    macro_f1_list.append(macro_f1)
    micro_f1_list.append(micro_f1)
 
if 'amazon_photo_network' in args.df_nodes_embedding_filepath:
    network_name = 'amazon'
elif 'cora_network' in args.df_nodes_embedding_filepath:
    network_name = 'cora'
else:
    network_name = 'lineagew'
print("node classification for {} network".format(network_name))
print("node embedding file: {}".format(args.df_nodes_embedding_filepath))
print("number of repetitions for stratified sampling: {}".format(args.num_repetitions))
print("average eval macro f1 score: {}".format(np.mean(macro_f1_list)))
print("average eval micro f1 score: {}".format(np.mean(micro_f1_list)))


