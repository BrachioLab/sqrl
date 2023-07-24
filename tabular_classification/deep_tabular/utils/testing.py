""" testing.py
    Utilities for testing models
    Developed for Tabular-Transfer-Learning project
    March 2022
    Some functionality adopted from https://github.com/Yura52/rtdl
"""

import torch
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from .test_time_adaptation import evaluate_rule_violations_backup
import logging
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from rule_processing.process_rules import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Subset
from rule_processing.dataset_for_sampling import *
import pandas as pd
# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def evaluate_model(net, loaders, task, device):
    scores = []
    for loader in loaders:
        score = test_default(net, loader, task, device)
        scores.append(score)
    return scores

def test_rule_violations(net, testloader, device, task, normalizer):
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(testloader, leave=False)):
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            elif task == "regression":
                predicted = outputs
            violation_count, violation_loss = evaluate_rule_violations(inputs_num, predicted, normalizer)
            # loss = compute_rule_loss(inputs_num, outputs, normalizer, criterion)
            total_violation_count += violation_count
            total_violation_loss += violation_loss
            # if loss.cpu().item() > 0:
            

        print("total violation count::", total_violation_count)
        print("total_violation_loss::", total_violation_loss)


def eval_test_rule_violations(net, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, task, device):
    net.eval()
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        
        # for item in tqdm(testloader):
        for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(testloader, leave=False)):
            if type(inputs_num) is list:
                inputs_num, inputs_cat, targets = inputs_num[0].to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                        inputs_cat if inputs_cat.nelement() != 0 else None
            else:
                inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                        inputs_cat if inputs_cat.nelement() != 0 else None
            # image, target, path, df = item
            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            elif task == "regression":
                predicted = outputs
            
            violation_loss, violation_count = evaluate_rule_violations(df, predicted, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            # loss = compute_rule_loss(inputs_num, outputs, normalizer, criterion)
            total_violation_count += violation_count
            if violation_loss > 0:
                total_violation_loss += violation_loss.cpu().item()
            # if loss.cpu().item() > 0:
            # del image, pred, violation_loss

        logging.info("total violation count::%d", total_violation_count)
        logging.info("total_violation_loss::%f", total_violation_loss)

    net.train()

    return total_violation_count, total_violation_loss

def perform_qualitative_studies(output_dir, net, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, task, device, test_bz):
    net.eval()
    
    dataset_for_sampling = Dataset_for_sampling(len(testloader.dataset))
    sampler = RandomSampler(dataset_for_sampling, replacement=True, num_samples=len(dataset_for_sampling)*20)
    dataset_for_sampling_loader = DataLoader(dataset_for_sampling, collate_fn = Dataset_for_sampling.collate_fn,sampler=sampler, batch_size=test_bz)
    total_violation_per_mb = {"mb_idx":[], "violation_count":[]}
    model_pred_ls = []
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        
        # for item in tqdm(testloader):
        all_meta_class_pred_rule_label_mappings = dict()
        df_ls = []
        for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(testloader, leave=False)):
            if type(inputs_num) is list:
                inputs_num, inputs_cat, targets = inputs_num[0].to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                        inputs_cat if inputs_cat.nelement() != 0 else None
            else:
                inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                        inputs_cat if inputs_cat.nelement() != 0 else None
            # image, target, path, df = item
            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            elif task == "regression":
                predicted = outputs
            meta_class_pred_rule_label_mappings = obtain_rule_evaluations(df, predicted, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            merge_meta_class_pred_rule_label_mappings(all_meta_class_pred_rule_label_mappings, meta_class_pred_rule_label_mappings)
            df_ls.append(df)
            model_pred_ls.append(predicted.view(-1).detach().cpu())
            # violation_loss, violation_count = evaluate_rule_violations(df, predicted, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            # loss = compute_rule_loss(inputs_num, outputs, normalizer, criterion)
        pred_label_df_dict = output_rule_predictions(all_meta_class_pred_rule_label_mappings, meta_class_pred_boolean_mappings)
        full_df = pd.concat(df_ls)
        model_pred_array = torch.cat(model_pred_ls).numpy()
        pred_label_df_dict["sample_ids"] = full_df.index
        pred_label_df_dict["model_pred_label"] = model_pred_array
        pred_label_df_dict = pd.DataFrame(pred_label_df_dict)
        pred_label_df_dict = pred_label_df_dict.set_index("sample_ids")
        pred_label_df_dict["violation_count"] = np.sum(np.array(pred_label_df_dict) == 0, axis=1)
        pred_label_df_dict["satisfy_count"] = np.sum(np.array(pred_label_df_dict) == 1, axis=1)
        pred_label_df_dict.to_csv(os.path.join(output_dir, "pred_label_df.csv"))
        full_df.to_csv(os.path.join(output_dir, "gt_df.csv"))
        mb_id = 0
        
        for sample_ids in tqdm(dataset_for_sampling_loader):
            sub_df = full_df.iloc[sample_ids]
            rule_loss, rule_violations, meta_class_rule_score_on_mb_mappings = post_eval_rule_f1_scores(sample_ids, all_meta_class_pred_rule_label_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, return_details=True)
            total_violation_count += rule_violations
            if rule_loss > 0:
                total_violation_loss += rule_loss#.cpu().item()
                
            detailed_df_f1_numbers = construct_detailed_df_output(meta_class_rule_score_on_mb_mappings, meta_class_pred_boolean_mappings)
            detailed_df_data = pd.DataFrame(detailed_df_f1_numbers)
            sub_df.to_csv(os.path.join(output_dir, "mb_dataframe_" + str(mb_id) + ".csv"))
            detailed_df_data.to_csv(os.path.join(output_dir, "detailed_f1_" + str(mb_id) + ".csv"))
            mini_batch_violations = sum(detailed_df_data["violation"])
            total_violation_per_mb["mb_idx"].append(mb_id)
            total_violation_per_mb["violation_count"].append(mini_batch_violations)
            mb_id += 1
            # if loss.cpu().item() > 0:
            # del image, pred, violation_loss
        total_violation_per_mb = pd.DataFrame(total_violation_per_mb)
        total_violation_per_mb.to_csv(os.path.join(output_dir, "total_violation_per_mb.csv"))
        logging.info("total violation count::%d", total_violation_count)
        logging.info("total_violation_loss::%f", total_violation_loss)

    net.train()

    return total_violation_count, total_violation_loss

def test_default(net, testloader, task, device):
    net.eval()
    targets_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(testloader, leave=False)):
            if type(inputs_num) is list:
                inputs_num, inputs_cat, targets = inputs_num[0].to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None
            else:
                inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
                inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = outputs
            elif task == "regression":
                predicted = outputs
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predicted.cpu().tolist())

    if task == "multiclass":
        accuracy = accuracy_score(targets_all, predictions_all)
        balanced_accuracy = balanced_accuracy_score(targets_all, predictions_all, adjusted=False)
        balanced_accuracy_adjusted = balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        scores = {"score": accuracy,
                  "accuracy": accuracy,
                  "balanced_accuracy": balanced_accuracy,
                  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
    elif task == "regression":
        rmse = mean_squared_error(targets_all, predictions_all, squared=False)
        scores = {"score": -rmse,
                  "rmse": -rmse}
    elif task == "binclass":
        roc_auc = roc_auc_score(targets_all, predictions_all)
        accuracy = accuracy_score(np.array(targets_all).reshape(-1), np.array(torch.sigmoid(torch.tensor(predictions_all)) > 0.5).reshape(-1))
        scores = {"score": roc_auc,"accuracy": accuracy,
                  "roc_auc": roc_auc}
    return scores


def evaluate_backbone(embedders, backbone, heads, loaders, tasks, device):
    scores = {}
    for k in loaders.keys():
        score = evaluate_backbone_one_dataset(embedders[k], backbone, heads[k], loaders[k], tasks[k], device)
        scores[k] = score
    return scores


def evaluate_backbone_one_dataset(embedder, backbone, head, testloader, task, device):
    embedder.eval()
    backbone.eval()
    head.eval()
    targets_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs_num, inputs_cat, targets,_) in enumerate(tqdm(testloader, leave=False)):
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            embedding = embedder(inputs_num, inputs_cat)
            features = backbone(embedding)
            outputs = head(features)

            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = outputs
            elif task == "regression":
                predicted = outputs
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predicted.cpu().tolist())

    if task == "multiclass":
        accuracy = accuracy_score(targets_all, predictions_all)
        balanced_accuracy = balanced_accuracy_score(targets_all, predictions_all, adjusted=False)
        balanced_accuracy_adjusted = balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        scores = {"score": accuracy,
                  "accuracy": accuracy,
                  "balanced_accuracy": balanced_accuracy,
                  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
    elif task == "regression":
        rmse = mean_squared_error(targets_all, predictions_all, squared=False)
        scores = {"score": -rmse,
                  "rmse": -rmse}
    elif task == "binclass":
        roc_auc = roc_auc_score(targets_all, predictions_all)
        scores = {"score": roc_auc,
                  "roc_auc": roc_auc}
    return scores
