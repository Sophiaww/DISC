import time
import torch
import dgl
import os
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader as torch_dataloader
from dgl.dataloading import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import dgl.function as fn
import logging
import pickle
import os


def get_dataset(name: str, raw_dir: str, to_homo: bool = False, random_state: int = 717):
    if name == 'yelp':
        yelp_data = dgl.data.FraudYelpDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = yelp_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)

    elif name == 'amazon':
        amazon_data = dgl.data.FraudAmazonDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = amazon_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)

    # elif name == 'tsocial':
    #     t_social, _ = load_graphs(os.path.join(raw_dir, 'tsocial'))
    #     graph = t_social[0]
    #     graph.ndata['feature'] = graph.ndata['feature'].float()

    # elif name == 'tfinance':
    #     t_finance, _ = load_graphs(os.path.join(raw_dir, 'tfinance'))
    #     graph = t_finance[0]
    #     graph.ndata['label'] = graph.ndata['label'].argmax(1)
    #     graph.ndata['feature'] = graph.ndata['feature'].float()

    elif name == 'questions':
        # Load graph from data/questions
        questions_graph, _ = load_graphs(os.path.join(raw_dir, 'questions'))
        graph = questions_graph[0]

        # assume feature is float
        graph.ndata['feature'] = graph.ndata['feature'].float()

        # 如果 label 是 one-hot，则自动 argmax
        if graph.ndata['label'].dim() > 1:
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        # 可选同质化
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'])
            graph = dgl.add_self_loop(graph)

    elif name == 'reddit':
        questions_graph, _ = load_graphs(os.path.join(raw_dir, 'reddit'))
        graph = questions_graph[0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        if graph.ndata['label'].dim() > 1:
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

    elif name == 'tolokers':
        questions_graph, _ = load_graphs(os.path.join(raw_dir, 'tolokers'))
        graph = questions_graph[0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        if graph.ndata['label'].dim() > 1:
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

    elif name == 'weibo':
        questions_graph, _ = load_graphs(os.path.join(raw_dir, 'weibo'))
        graph = questions_graph[0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        if graph.ndata['label'].dim() > 1:
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

    else:
        raise

    return graph


def get_index_loader_test(name: str, batch_size: int, unlabel_ratio: int = 1, training_ratio: float = -1,
                          shuffle_train: bool = True, to_homo: bool = False):
    assert name in ['yelp', 'amazon', 'questions', 'reddit', 'tolokers', 'weibo'], 'Invalid dataset name'

    graph = get_dataset(name, 'data/', to_homo=to_homo, random_state=7537)

    index = np.arange(graph.num_nodes())
    labels = graph.ndata['label']
    if name == 'amazon':
        index = np.arange(3305, graph.num_nodes())

    train_nids, valid_test_nids = train_test_split(index, stratify=labels[index],
                                                   train_size=training_ratio / 100., random_state=2, shuffle=True)
    valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids],
                                             test_size=0.67, random_state=2, shuffle=True)

    train_mask = torch.zeros_like(labels).bool()
    val_mask = torch.zeros_like(labels).bool()
    test_mask = torch.zeros_like(labels).bool()
    # train_nids = torch.from_numpy(train_nids).to(torch.int32)
    # valid_nids = torch.from_numpy(valid_nids).to(torch.int32)
    # test_nids = torch.from_numpy(test_nids).to(torch.int32)

    train_mask[train_nids] = 1
    val_mask[valid_nids] = 1
    test_mask[test_nids] = 1

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    labeled_nids = train_nids
    unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])

    power = 10 if name == 'tfinance' else 16

    valid_loader = torch_dataloader(valid_nids, batch_size=2 ** power, shuffle=False, drop_last=False, num_workers=4)
    test_loader = torch_dataloader(test_nids, batch_size=2 ** power, shuffle=False, drop_last=False, num_workers=4)
    labeled_loader = torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True,
                                      num_workers=0)
    unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train,
                                        drop_last=True, num_workers=0)

    return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader

