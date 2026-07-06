import argparse
import sys
import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules.data_loader import get_index_loader_test
from models import simpleGNN_MR
import modules.mod_utls as m_utls
# HighConfidenceGraphExtractor is no longer needed for threshold method
# from modules.high_conf_selector import HighConfidenceGraphExtractor
from modules.loss import *
from modules.evaluation import eval_pred
from modules.aux_mod import fixed_augmentation
from sklearn.metrics import f1_score
from modules.conv_mod import CustomLinear
from modules.mr_conv_mod import build_mlp
import numpy as np
from numpy import random
import math
import pandas as pd
from functools import partial
import dgl
import warnings
import wandb
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


warnings.filterwarnings("ignore")


def create_model(args, e_ts):
    if args['model'] == 'backbone':
        tmp_model = simpleGNN_MR(in_feats=args['node-in-dim'], hidden_feats=args['hidden-dim'],
                                 out_feats=args['node-out-dim'],
                                 num_layers=args['num-layers'], e_types=e_ts, input_drop=args['input-drop'],
                                 hidden_drop=args['hidden-drop'],
                                 mlp_drop=args['mlp-drop'], mlp12_dim=args['mlp12-dim'], mlp3_dim=args['mlp3-dim'],
                                 bn_type=args['bn-type'])
    else:
        raise
    tmp_model.to(args['device'])

    return tmp_model

def UDA_train_epoch(epoch, model, loss_func, graph, label_loader, unlabel_loader, optimizer, sampler, args, loss_type='rbs'):
    model.train()
    num_iters = args['train-iterations']

    sup_loss_sum = 0.0
    unsup_loss_sum = 0.0
    total_loss_sum = 0.0

    unlabel_loader_iter = iter(unlabel_loader)
    label_loader_iter = iter(label_loader)

    # --- 数据收集容器 (用于可视化) ---
    all_embeddings = []
    all_labels = []
    collect_for_viz = (epoch == 0 or (epoch + 1) % 20 == 0)
    # -------------------------------

    for idx in range(num_iters):
        try:
            label_idx = label_loader_iter.__next__()
        except:
            label_loader_iter = iter(label_loader)
            label_idx = label_loader_iter.__next__()
        try:
            unlabel_idx = unlabel_loader_iter.__next__()
        except:
            unlabel_loader_iter = iter(unlabel_loader)
            unlabel_idx = unlabel_loader_iter.__next__()

        unsup_loss = torch.tensor(0.0, device=args['device'], requires_grad=True)

        # -------------------------------------------------------------
        # 1. 阈值法伪标签生成 (Threshold-based Pseudo-labeling)
        # -------------------------------------------------------------
        u_mask = None
        pseudo_labels = None

        # 确保在 warm-up 之后才开始自训练
        if epoch > args['trainable-warm-up']:
            model.eval()
            with torch.no_grad():
                # 生成无标签数据的 view
                _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='none')

                # 前向传播获取 logits
                weak_inter_results = model(u_blocks, update_bn=False, return_logits=True)
                weak_h = torch.stack(weak_inter_results, dim=1)
                weak_h = weak_h.reshape(weak_h.shape[0], -1)
                weak_logits = model.proj_out(weak_h)

                u_pred_weak_log = weak_logits.log_softmax(dim=-1)
                u_pred_weak = u_pred_weak_log.exp()[:, 1]  # 获取 class 1 的概率

                # === 使用阈值策略 ===
                pseudo_labels = torch.ones_like(u_pred_weak).long()  # 初始化占位，具体值由 mask 决定

                # 假设 args 中包含 'normal-th' (e.g., 50) 和 'fraud-th' (e.g., 80)
                # 转换为概率 0.5 和 0.8
                neg_tar = (u_pred_weak <= (args['normal-th'] / 100.)).bool()
                pos_tar = (u_pred_weak >= (args['fraud-th'] / 100.)).bool()

                pseudo_labels[neg_tar] = 0
                pseudo_labels[pos_tar] = 1

                # 只有非常自信的（小于下阈值 或 大于上阈值）才参与训练
                u_mask = torch.logical_or(neg_tar, pos_tar)
                # ============================

        # -------------------------------------------------------------
        # 2. 训练步骤
        # -------------------------------------------------------------
        model.train()

        # 重新为无标签数据生成计算图 (Forward with gradients)
        # 注意：这里需要重新 sample 或者复用 u_blocks，只要确保是在 model.train() 下运行
        _, _, u_blocks_train = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='none')
        weak_inter_results_train = model(u_blocks_train, update_bn=False, return_logits=True)
        weak_h_train = torch.stack(weak_inter_results_train, dim=1).reshape(weak_inter_results_train[0].shape[0], -1)
        weak_logits_train = model.proj_out(weak_h_train)

        # --- 收集数据用于可视化 ---
        # if collect_for_viz:
        #     batch_labels = graph.ndata['label'][unlabel_idx].cpu().numpy()
        #     all_embeddings.append(weak_h_train.detach().cpu().numpy())
        #     all_labels.append(batch_labels)
        # # -----------------------

        # 计算无监督损失 (Unsupervised Loss)
        if u_mask is not None and u_mask.sum() > 0:
            u_logits_confident = weak_logits_train[u_mask]
            pred_label_confident = pseudo_labels[u_mask]

            # === 使用标准交叉熵 (Standard Cross Entropy) ===
            # unsup_loss = F.cross_entropy(u_logits_confident, pred_label_confident)
            # ========================================================
            # if loss_type == 'ce':
            #     # === 对比方案 A: 标准交叉熵 ===
            #     unsup_loss = F.cross_entropy(u_logits_confident, pred_label_confident)

            # elif loss_type == 'rbs':
            # === 对比方案 B: Robust Balanced Softmax ===
            u_pred_log = weak_logits_train.log_softmax(dim=-1)
            u_pred_log_confident = u_pred_log[u_mask]
            with torch.no_grad():

                class_num_list = torch.tensor(
                    [(pred_label_confident == i).sum().item() for i in range(2)]
                ).to(args['device'])

            # 调用 RBS Loss
            loss_val = model.RobustBalancedSoftmax(u_pred_log_confident, pred_label_confident, class_num_list)
            unsup_loss = torch.mean(loss_val)

        else:
            unsup_loss = torch.tensor(0.0, device=args['device'], requires_grad=True)

        # 有监督样本前向传播
        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(args['device']), sampler, aug_type='none')
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']

        sup_loss, _ = loss_func(s_pred, s_target)

        unsup_weight = args.get('unsup-weight', 1.0)
        loss = sup_loss + unsup_weight * unsup_loss + args['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sup_loss_sum += sup_loss.item()
        unsup_loss_sum += unsup_loss.item() if isinstance(unsup_loss, torch.Tensor) else unsup_loss
        total_loss_sum += loss.item()

    avg_sup_loss = sup_loss_sum / num_iters
    avg_unsup_loss = unsup_loss_sum / num_iters
    avg_total_loss = total_loss_sum / num_iters

    print(f"Epoch {epoch + 1} | Avg Total Loss: {avg_total_loss:.4f}")
    return avg_sup_loss, avg_unsup_loss, avg_total_loss


def get_model_pred(model, graph, data_loader, sampler, args):
    model.eval()

    pred_list = []
    target_list = []
    with torch.no_grad():
        for node_idx in data_loader:
            _, _, blocks = sampler.sample_blocks(graph, node_idx.to(args['device']))

            pred = model(blocks)
            target = blocks[-1].dstdata['label']

            pred_list.append(pred.detach())
            target_list.append(target.detach())
        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        pred_list = pred_list.exp()[:, 1]

    return pred_list, target_list


def val_epoch(epoch, model, graph, valid_loader, test_loader, sampler, args):
    valid_dict = {}
    valid_pred, valid_target = get_model_pred(model, graph, valid_loader, sampler, args)
    v_roc, v_pr, _, _, _, _, v_f1, v_thre = eval_pred(valid_pred, valid_target)
    valid_dict['auc-roc'] = v_roc
    valid_dict['auc-pr'] = v_pr
    valid_dict['marco f1'] = v_f1

    test_dict = {}
    test_pred, test_target = get_model_pred(model, graph, test_loader, sampler, args)
    t_roc, t_pr, _, _, _, _, _, _ = eval_pred(test_pred, test_target)
    test_dict['auc-roc'] = t_roc
    test_dict['auc-pr'] = t_pr

    test_pred = test_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    guessed_target = np.zeros_like(test_target)
    guessed_target[test_pred > v_thre] = 1
    t_f1 = f1_score(test_target, guessed_target, average='macro')
    test_dict['marco f1'] = t_f1

    return valid_dict, test_dict


def run_model(args):
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'],
                                                                                           batch_size=args[
                                                                                               'batch-size'],
                                                                                           unlabel_ratio=args[
                                                                                               'unlabel-ratio'],
                                                                                           training_ratio=args[
                                                                                               'training-ratio'],
                                                                                           shuffle_train=args[
                                                                                               'shuffle-train'],
                                                                                           to_homo=args['to-homo'])

    graph = graph.to(args['device'])

    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2

    my_model = create_model(args, graph.etypes)

    if args['optim'] == 'adam':
        optimizer = optim.Adam(my_model.parameters(), lr=float(args['lr']), weight_decay=0.0)
    elif args['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(my_model.parameters(), lr=float(args['lr']), weight_decay=0.0)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])

    # Extractor Removed: use simple thresholding now.
    # extractor = HighConfidenceGraphExtractor(...)

    task_loss = focal_loss
    best_val = sys.float_info.min

    print(f"Starting training with Threshold-based Pseudo-labeling + Standard CE")

    for epoch in range(args['epochs']):
        avg_sup, avg_unsup, avg_total = UDA_train_epoch(
            epoch, my_model, task_loss, graph, label_loader, unlabel_loader,
            optimizer, sampler, args)

        val_results, test_results = val_epoch(epoch, my_model, graph, valid_loader, test_loader, sampler, args)

        print(
            f"  Val AUC: {val_results['auc-roc']:.4f} | Test AUC: {test_results['auc-roc']:.4f} | Test F1: {test_results['marco f1']:.4f}")

        if val_results['auc-roc'] > best_val:
            best_val = val_results['auc-roc']
            test_in_best_val = test_results
            if args['store-model']:
                m_utls.store_model(my_model, args)

    print("==== Training finished ====")
    return list(test_in_best_val.values())


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs.')

    cfg = vars(parser.parse_args())

    args = get_config(cfg['config'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:%d' % (args['device']))
    else:
        args['device'] = torch.device('cpu')

    print(args)
    final_results = []
    seed = 123
    for r in range(cfg['runs']):
        fix_seed(seed=seed)
        final_results.append(run_model(args))

    final_results = np.array(final_results)
    mean_results = np.mean(final_results, axis=0)
    std_results = np.std(final_results, axis=0)

    print(f"Results for Threshold + Standard CE:")
    print(mean_results)
    print(std_results)
    print('total time: ', time.time() - start_time)