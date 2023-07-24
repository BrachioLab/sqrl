import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from rule_processing.dataset_for_sampling import Dataset_for_sampling
from torch.utils.data import DataLoader, RandomSampler
import torch
from tqdm import tqdm

import pandas as pd
from rule_processing.process_rules import obtain_rule_evaluations, merge_meta_class_pred_rule_label_mappings, post_eval_rule_f1_scores, output_rule_predictions, construct_detailed_df_output

import logging
import numpy as np

def get_meta_class_reverse_mappings(meta_class_mappings):
    reverse_mappings = dict()
    for meta_class in meta_class_mappings:
        reverse_mappings[meta_class_mappings[meta_class]] = meta_class
    return reverse_mappings

def check_correctness(pred, meta_class_pred_rule_label_mappings, meta_reverse_label_mappings):
    pred_labels = pred.argmax(-1).tolist()
    for metaclass in meta_class_pred_rule_label_mappings:
        pred_rule_labels = meta_class_pred_rule_label_mappings[metaclass]
        pred_rule_labels_tensors = np.stack(pred_rule_labels, axis=0)
        for idx in range(pred_rule_labels_tensors.shape[1]):
        # for idx in range(len(pred_rule_labels)):
            if np.sum(np.array(pred_rule_labels_tensors[:,idx]) > 0) > 0:
                assert meta_reverse_label_mappings[pred_labels[idx]] == metaclass

def perform_qualitative_studies(output_dir, net, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, test_bz):
    net.eval()
    meta_class_reverse_mappings= get_meta_class_reverse_mappings(meta_class_mappings)
    dataset_for_sampling = Dataset_for_sampling(len(testloader.dataset))
    sampler = RandomSampler(dataset_for_sampling, replacement=True, num_samples=len(dataset_for_sampling)*5)
    dataset_for_sampling_loader = DataLoader(dataset_for_sampling, collate_fn = Dataset_for_sampling.collate_fn,sampler=sampler, batch_size=test_bz)
    model_pred_ls = []
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        total_violation_per_mb = {"mb_idx":[], "violation_count":[]}
        # for item in tqdm(testloader):
        all_meta_class_pred_rule_label_mappings = dict()
        df_ls = []
        # for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(testloader, leave=False)):
        for item in tqdm(testloader):
            image, target, path, df = item
            if type(image) is list:
                image = image[0]
            if torch.cuda.is_available():
                image = image.cuda()
            outputs = net(image)

            # predicted = outputs
            pred = torch.softmax(outputs, dim=-1)
            meta_class_pred_rule_label_mappings = obtain_rule_evaluations(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            
            check_correctness(pred, meta_class_pred_rule_label_mappings, meta_class_reverse_mappings)
            
            
            merge_meta_class_pred_rule_label_mappings(all_meta_class_pred_rule_label_mappings, meta_class_pred_rule_label_mappings)
            df["path"] = path
            df_ls.append(df)
            pred_labels = torch.argmax(pred, dim=-1)
            model_pred_ls.append(pred_labels.view(-1).cpu().detach())
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