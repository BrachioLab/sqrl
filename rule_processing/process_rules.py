import json

import torch

import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from rule_processing.sigmoidF1 import sigmoidF1

from sklearn.metrics import f1_score
import logging


# metaclass_name_id_mappings = {}
# "other(V0) :- !style(V0), !smaller(V0), pose(V0), minibatch(V0)"
def parse_single_rule(rule, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_rule_high_score_mappings, rule_lower_score, rule_higher_score):
    head, body = rule.split(":-")
    head = head.strip()
    body = body.strip()
    metaclass = head.split("(V0)")[0]
    predicate_ls = body.split(",")
    pred_boolean_mappings = dict()
    assert "minibatch" in predicate_ls[-1]
    del predicate_ls[-1]

    for pred in predicate_ls:
        pred = pred.strip()
        pred = pred.split("(V0)")[0]
        
        if pred.startswith("!"):
            attr_name = pred[1:]
            pred_boolean_mappings[attr_name] = 0
        else:
            attr_name = pred
            pred_boolean_mappings[attr_name] = 1

    # keep_rule = filter_rules_by_symbolic_conditions(pred_boolean_mappings)
    # if not keep_rule:
    #     return

    if metaclass not in meta_class_pred_boolean_mappings:
        meta_class_pred_boolean_mappings[metaclass] = []
        meta_class_rule_score_mappings[metaclass] = []
        meta_class_rule_high_score_mappings[metaclass] = []

    meta_class_pred_boolean_mappings[metaclass].append(pred_boolean_mappings)
    meta_class_rule_score_mappings[metaclass].append((rule_lower_score, rule_higher_score))
    meta_class_rule_high_score_mappings[metaclass].append(rule_lower_score)


def select_top_k_rules_per_class(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_rule_high_score_mappings, k =20):
    if meta_class_rule_high_score_mappings is None:
        meta_class_rule_high_score_mappings = dict()
        for metaclass in meta_class_rule_score_mappings:
            meta_class_rule_high_score_mappings[metaclass] = []
            for idx in range(len(meta_class_rule_score_mappings[metaclass])):
                meta_class_rule_high_score_mappings[metaclass].append(meta_class_rule_score_mappings[metaclass][idx][0])

    for metaclass in meta_class_rule_high_score_mappings:
        high_score_ls = meta_class_rule_high_score_mappings[metaclass]
        high_score_tensor = torch.tensor(high_score_ls)
        sorted_high_score_tensor, sorted_indices = torch.sort(high_score_tensor, descending=True)
        if len(sorted_indices) > k:
            top_k_indices = sorted_indices[0:k].tolist()
        else:
            top_k_indices = sorted_indices.tolist()
        meta_class_pred_boolean_mappings[metaclass] = [meta_class_pred_boolean_mappings[metaclass][idx] for idx in top_k_indices]
        meta_class_rule_score_mappings[metaclass] = [meta_class_rule_score_mappings[metaclass][idx] for idx in top_k_indices]
        meta_class_rule_high_score_mappings[metaclass] = [meta_class_rule_high_score_mappings[metaclass][idx] for idx in top_k_indices]


def merge_meta_class_pred_rule_label_mappings(all_meta_class_pred_rule_label_mappings, meta_class_pred_rule_label_mappings):
    if len(all_meta_class_pred_rule_label_mappings) <= 0:
        for meta_class in meta_class_pred_rule_label_mappings:
            all_meta_class_pred_rule_label_mappings[meta_class] = []
            all_meta_class_pred_rule_label_mappings[meta_class].extend(meta_class_pred_rule_label_mappings[meta_class])
            
    else:
        for meta_class in meta_class_pred_rule_label_mappings:
            for idx in range(len(meta_class_pred_rule_label_mappings[meta_class])):
                all_meta_class_pred_rule_label_mappings[meta_class][idx] = np.concatenate([all_meta_class_pred_rule_label_mappings[meta_class][idx], meta_class_pred_rule_label_mappings[meta_class][idx]])
                

def merge_meta_class_f1_score_ls_mappings(all_meta_class_pred_rule_label_mappings, meta_class_rule_score_on_mb_mappings):
    # if len(all_meta_class_pred_rule_label_mappings) <= 0:
    #     for meta_class in meta_class_rule_score_on_mb_mappings:
    #         all_meta_class_pred_rule_label_mappings[meta_class] = []
            # all_meta_class_pred_rule_label_mappings[meta_class].append(meta_class_rule_score_on_mb_mappings[meta_class][0])
            
    # else:
    for meta_class in meta_class_rule_score_on_mb_mappings:
        for idx in range(len(meta_class_rule_score_on_mb_mappings[meta_class])):
            if meta_class_rule_score_on_mb_mappings[meta_class][idx][0] >= 0:
                all_meta_class_pred_rule_label_mappings[meta_class][idx].append(meta_class_rule_score_on_mb_mappings[meta_class][idx][0])
                # all_meta_class_pred_rule_label_mappings[meta_class][idx] = np.concatenate([meta_class_rule_score_on_mb_mappings[meta_class][idx], all_meta_class_pred_rule_label_mappings[meta_class][idx]])


def parse_rule_file(rule_file, k =20, curr_dir=None):
    if curr_dir is None:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
    logging.info("rule file name::%s", os.path.join(curr_dir, rule_file))
    with open(os.path.join(curr_dir, rule_file)) as f:
        rule_json_ls = list(f)
    
    all_rule_obj = dict()
    all_rule_obj_size = dict()
    for rule_json in rule_json_ls:
        rule_obj = json.loads(rule_json)
        all_rule_obj[rule_obj["rule"]] = rule_obj["bounds"]
        all_rule_obj_size[rule_obj["rule"]] = rule_obj["sample_size"]
    meta_class_pred_boolean_mappings = dict()
    meta_class_rule_score_mappings = dict()
    meta_class_rule_high_score_mappings = dict()
    for rule in all_rule_obj:
        if len(all_rule_obj[rule]) < 2:
            continue
        rule_lower_score, rule_higher_score = all_rule_obj[rule][0], all_rule_obj[rule][1]

        parse_single_rule(rule, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_rule_high_score_mappings, rule_lower_score, rule_higher_score)
    if k > 0:
        select_top_k_rules_per_class(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_rule_high_score_mappings, k=k)
    return meta_class_pred_boolean_mappings, meta_class_rule_score_mappings

def apply_single_rule(df, pred, meta_class, pred_boolean_mappings, F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1):
    # selected_df = df[df["metaclass"] == meta_class]
    sample_id_boolean = None#(df["metaclass"] == meta_class)
    for attr in pred_boolean_mappings:
        expected_attr_val = pred_boolean_mappings[attr]
        curr_sample_id_boolean = (df[attr] == expected_attr_val)
        if sample_id_boolean is None:
            sample_id_boolean = np.array(curr_sample_id_boolean)
        else:
            sample_id_boolean = np.logical_and(sample_id_boolean , np.array(curr_sample_id_boolean))
    sample_id_boolean = np.array(sample_id_boolean)

    if np.sum(sample_id_boolean) <= 0:
        return 0

    selected_pred = pred[sample_id_boolean][:, meta_class_mappings[meta_class]]

    # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    expected_pred = torch.ones_like(selected_pred)
    # if expected_label == 1:
    F1_score = sigmoidF1(selected_pred, expected_pred)
    # else:
    #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])

    if F1_score >= bounds[0] and F1_score <= bounds[1]:
        return 0

    if F1_score < bounds[0]:
        loss = torch.abs(F1_score - bounds[0])
    elif F1_score > bounds[1]:
        loss = torch.abs(F1_score - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    del expected_pred
    return loss



def do_apply_single_rule_per_sample_eval(df, pred, meta_class, rule):
    # selected_df = df[df["metaclass"] == meta_class]
    # sample_id_boolean = None#(df["metaclass"] == meta_class)
    df_copy = df.copy()
    
    sample_id_boolean = np.ones(len(df)).astype(bool)
    
    for attr in rule:
        bound = rule[attr]
        if len(bound) <= 0:
            return 0          
        sample_id_boolean = sample_id_boolean & ((df_copy[attr] > bound[0]) & (df_copy[attr] < bound[1]))
    
    
    sample_id_boolean = torch.from_numpy(np.array(sample_id_boolean)).to(pred.device)

    sample_id_boolean = sample_id_boolean & (torch.argmax(pred, dim=-1) != meta_class)

    # for attr in pred_boolean_mappings:
    #     expected_attr_val = pred_boolean_mappings[attr]
    #     curr_sample_id_boolean = (df[attr] == expected_attr_val)
    #     if sample_id_boolean is None:
    #         sample_id_boolean = curr_sample_id_boolean
    #     else:
    #         sample_id_boolean = (sample_id_boolean & curr_sample_id_boolean)
    # sample_id_boolean = np.array(sample_id_boolean)

    


    # expected_labels[:, meta_class]=0

    # loss = -torch.mean((torch.log_softmax(selected_pred, dim=-1)*expected_labels))

    return torch.sum(sample_id_boolean)


def do_apply_single_rule_per_sample_eval_full(df, pred, meta_class, rule):
    # selected_df = df[df["metaclass"] == meta_class]
    # sample_id_boolean = None#(df["metaclass"] == meta_class)
    df_copy = df.copy()
    
    sample_id_boolean = np.ones(len(df)).astype(bool)
    
    for attr in rule:
        bound = rule[attr]
        if len(bound) <= 0:
            return 0          
        sample_id_boolean = sample_id_boolean & ((df_copy[attr] > bound[0]) & (df_copy[attr] < bound[1]))
    
    
    sample_id_boolean = torch.from_numpy(np.array(sample_id_boolean)).to(pred.device)

    sample_id_boolean = sample_id_boolean & (torch.argmax(pred, dim=-1) != meta_class)

    # for attr in pred_boolean_mappings:
    #     expected_attr_val = pred_boolean_mappings[attr]
    #     curr_sample_id_boolean = (df[attr] == expected_attr_val)
    #     if sample_id_boolean is None:
    #         sample_id_boolean = curr_sample_id_boolean
    #     else:
    #         sample_id_boolean = (sample_id_boolean & curr_sample_id_boolean)
    # sample_id_boolean = np.array(sample_id_boolean)

    


    # expected_labels[:, meta_class]=0

    # loss = -torch.mean((torch.log_softmax(selected_pred, dim=-1)*expected_labels))

    return sample_id_boolean


def do_apply_single_rule_per_sample(df, pred, meta_class, rule, criterion):
    # selected_df = df[df["metaclass"] == meta_class]
    # sample_id_boolean = None#(df["metaclass"] == meta_class)
    df_copy = df.copy()
    
    sample_id_boolean = np.ones(len(df)).astype(bool)
    
    for attr in rule:
        bound = rule[attr]    
        if len(bound) <= 0:
            return 0        
        sample_id_boolean = sample_id_boolean & ((df_copy[attr] > bound[0]) & (df_copy[attr] < bound[1]))
    
    
    sample_id_boolean = torch.from_numpy(np.array(sample_id_boolean)).to(pred.device)

    sample_id_boolean = sample_id_boolean & (torch.argmax(pred, dim=-1) != meta_class)

    # for attr in pred_boolean_mappings:
    #     expected_attr_val = pred_boolean_mappings[attr]
    #     curr_sample_id_boolean = (df[attr] == expected_attr_val)
    #     if sample_id_boolean is None:
    #         sample_id_boolean = curr_sample_id_boolean
    #     else:
    #         sample_id_boolean = (sample_id_boolean & curr_sample_id_boolean)
    # sample_id_boolean = np.array(sample_id_boolean)

    

    selected_pred = pred[sample_id_boolean]
    
    if len(selected_pred) <= 0:
        return 0
    
    selected_pred = torch.softmax(selected_pred, dim=-1)
    
    loss = torch.max(selected_pred, dim = -1)[0] - selected_pred[:,meta_class]
    
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # 
    # loss = criterion(selected_pred, torch.ones(len(selected_pred),device=selected_pred.device, dtype=torch.long)*meta_class)

    # expected_labels = torch.ones((len(selected_pred), selected_pred.shape[-1]),device=selected_pred.device)

    # expected_labels[:, meta_class]=0

    # loss = -torch.mean((torch.log_softmax(selected_pred, dim=-1)*expected_labels))

    return loss.mean()

    # # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    # expected_pred = torch.ones_like(selected_pred)
    # # if expected_label == 1:
    # F1_score = sigmoidF1(selected_pred, expected_pred)
    # # else:
    # #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    # bounds = torch.tensor([F1_score_low, F1_score_high])

    # if F1_score >= bounds[0] and F1_score <= bounds[1]:
    #     return 0

    # if F1_score < bounds[0]:
    #     loss = torch.abs(F1_score - bounds[0])
    # elif F1_score > bounds[1]:
    #     loss = torch.abs(F1_score - bounds[1])
    # loss = torch.clamp(loss, min=0.0, max=1.0)
    # # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # # loss = torch.tanh(loss)
    # del expected_pred
    # return loss


def apply_single_rule_per_sample(df, pred, meta_class, pred_boolean_mappings, F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1):
    # selected_df = df[df["metaclass"] == meta_class]
    sample_id_boolean = None#(df["metaclass"] == meta_class)
    for attr in pred_boolean_mappings:
        expected_attr_val = pred_boolean_mappings[attr]
        curr_sample_id_boolean = (df[attr] == expected_attr_val)
        if sample_id_boolean is None:
            sample_id_boolean = np.array(curr_sample_id_boolean)
        else:
            sample_id_boolean = np.logical_and(sample_id_boolean, np.array(curr_sample_id_boolean))
    sample_id_boolean = np.array(sample_id_boolean)

    if np.sum(sample_id_boolean) <= 0:
        return 0

    selected_pred = pred[sample_id_boolean][:, meta_class_mappings[meta_class]]

    # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    expected_pred = torch.ones_like(selected_pred)
    # if expected_label == 1:
    F1_score = sigmoidF1(selected_pred, expected_pred)
    # else:
    #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])

    if F1_score >= bounds[0] and F1_score <= bounds[1]:
        return 0

    if F1_score < bounds[0]:
        loss = torch.abs(F1_score - bounds[0])
    elif F1_score > bounds[1]:
        loss = torch.abs(F1_score - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    del expected_pred
    return loss


def apply_single_rule_eval(df, pred, meta_class, pred_boolean_mappings, F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1):
    # selected_df = df[df["metaclass"] == meta_class]
    sample_id_boolean = None#(df["metaclass"] == meta_class)
    for attr in pred_boolean_mappings:
        expected_attr_val = pred_boolean_mappings[attr]
        curr_sample_id_boolean = (df[attr] == expected_attr_val)
        if sample_id_boolean is None:
            sample_id_boolean = np.array(curr_sample_id_boolean)
        else:
            sample_id_boolean = np.logical_and(sample_id_boolean , np.array(curr_sample_id_boolean))

    if len(pred.shape) > 1:
        pred_labels = torch.argmax(pred, dim=-1)
        pred_labels = pred_labels.view(-1)
    else:
        pred_labels = pred
    sample_id_boolean = np.array(sample_id_boolean)

    if np.sum(sample_id_boolean) <= 0:
        return 0, 0

    selected_pred = (pred_labels[sample_id_boolean] == meta_class_mappings[meta_class]).long().view(-1)

    # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    expected_pred = torch.ones_like(selected_pred)
    # if expected_label == 1:
    # F1_score = sigmoidF1(selected_pred, expected_pred)
    F1_score = f1_score(selected_pred.cpu().numpy(), expected_pred.cpu().numpy())
    # else:
    #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])


    violations = 0
    if F1_score < bounds[0]:
        loss = torch.abs(F1_score - bounds[0])
        violations = 1
        loss = torch.clamp(loss, min=0.0, max=1.0)
    elif F1_score > bounds[1]:
        loss = torch.abs(F1_score - bounds[1])
        violations = 1
        loss = torch.clamp(loss, min=0.0, max=1.0)
    else:
        loss = 0
    
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return loss, violations

def eval_rule_single(pred_labels, F1_score_low, F1_score_high):
    if np.sum(pred_labels >= 0) <= 0:
        return 0, 0, -1
    valid_pred_labels = pred_labels[pred_labels >= 0]
    expected_pred = np.ones_like(valid_pred_labels)
    F1_score = f1_score(valid_pred_labels, expected_pred)
    bounds = np.array([F1_score_low, F1_score_high])
    

    violations = 0
    if F1_score < bounds[0]:
        loss = np.abs(F1_score - bounds[0])
        violations = 1
        # loss = np.clamp(loss, min=0.0, max=1.0)
    elif F1_score > bounds[1]:
        loss = np.abs(F1_score - bounds[1])
        violations = 1
        # loss = torch.clamp(loss, min=0.0, max=1.0)
    else:
        loss = 0
    if loss > 1:
        loss = 1
    elif loss < 0:
        loss = 0
    
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return loss, violations, F1_score


def obtain_rule_eval(df, pred, meta_class, pred_boolean_mappings, meta_class_mappings):
    # selected_df = df[df["metaclass"] == meta_class]
    sample_id_boolean = None#(df["metaclass"] == meta_class)
    for attr in pred_boolean_mappings:
        expected_attr_val = pred_boolean_mappings[attr]
        curr_sample_id_boolean = (df[attr] == expected_attr_val)
        if sample_id_boolean is None:
            sample_id_boolean = np.array(curr_sample_id_boolean)
        else:
            sample_id_boolean = np.logical_and(sample_id_boolean, np.array(curr_sample_id_boolean))

    if len(pred.shape) > 1:
        pred_labels = torch.argmax(pred, dim=-1)
        pred_labels = pred_labels.view(-1)
    else:
        pred_labels = pred
    sample_id_boolean = np.array(sample_id_boolean)

    rule_eval_res = torch.ones_like(pred_labels)*(-1)

    rule_eval_res[sample_id_boolean] = 0
    # rule_eval_res[(pred_labels[sample_id_boolean] == meta_class_mappings[meta_class])] = 1
    rule_eval_res[sample_id_boolean] = (pred_labels[sample_id_boolean] == meta_class_mappings[meta_class]).long()


    # selected_pred = (pred_labels[sample_id_boolean] == meta_class_mappings[meta_class]).long().view(-1)
    return rule_eval_res

    # # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    # expected_pred = torch.ones_like(selected_pred)
    # # if expected_label == 1:
    # # F1_score = sigmoidF1(selected_pred, expected_pred)
    # F1_score = f1_score(selected_pred.cpu().numpy(), expected_pred.cpu().numpy())
    # # else:
    # #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    # bounds = torch.tensor([F1_score_low, F1_score_high])


    # violations = 0
    # if F1_score < bounds[0]:
    #     loss = torch.abs(F1_score - bounds[0])
    #     violations = 1
    #     loss = torch.clamp(loss, min=0.0, max=1.0)
    # elif F1_score > bounds[1]:
    #     loss = torch.abs(F1_score - bounds[1])
    #     violations = 1
    #     loss = torch.clamp(loss, min=0.0, max=1.0)
    # else:
    #     loss = 0
    
    # # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # # loss = torch.tanh(loss)
    
    # return loss, violations


def validate_single_rule_eval(df, labels, meta_class, pred_boolean_mappings, F1_score_low, F1_score_high, meta_class_mappings):
    # selected_df = df[df["metaclass"] == meta_class]
    sample_id_boolean = None#(df["metaclass"] == meta_class)
    for attr in pred_boolean_mappings:
        expected_attr_val = pred_boolean_mappings[attr]
        curr_sample_id_boolean = np.array(df[attr] == expected_attr_val)
        if sample_id_boolean is None:
            sample_id_boolean = np.array(curr_sample_id_boolean)
        else:
            sample_id_boolean = np.logical_and(sample_id_boolean, np.array(curr_sample_id_boolean))

    # pred_labels = torch.argmax(pred, dim=-1)
    pred_labels = labels.view(-1)
    sample_id_boolean = np.array(sample_id_boolean)

    if np.sum(sample_id_boolean) <= 0:
        return 0, 0, -1
        # return -1
    if meta_class_mappings is not None:
        selected_pred = (pred_labels[sample_id_boolean] == meta_class_mappings[meta_class]).long().view(-1)
    else:
        selected_pred = (pred_labels[sample_id_boolean] == meta_class).long().view(-1)

    # full_selected_pred = torch.cat([1-selected_pred, selected_pred], dim = 1)
    expected_pred = torch.ones_like(selected_pred)
    # if expected_label == 1:
    F1_score = f1_score(selected_pred.cpu().numpy(), expected_pred.cpu().numpy())
    # else:
    #     F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])


    violations = 0
    if F1_score < bounds[0]:
        loss = torch.abs(F1_score - bounds[0])
        violations = 1
    elif F1_score > bounds[1]:
        loss = torch.abs(F1_score - bounds[1])
        violations = 1
    else:
        loss = torch.tensor(0)
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return loss, violations, F1_score



def apply_rules_minibatch(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, criterion):
    total_rule_loss = 0
    total_count = 0
    for meta_class in meta_class_pred_boolean_mappings:
        pred_boolean_mappings_ls = meta_class_pred_boolean_mappings[meta_class]
        rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]

        for idx in range(len(pred_boolean_mappings_ls)):
            F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            curr_loss = apply_single_rule(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, criterion)
            total_rule_loss += curr_loss
            if curr_loss > 0:
                total_count += 1
    if total_count > 0:
        return total_rule_loss/total_count
    else:
        return total_rule_loss

def apply_rules_single_per_sample(df, pred, meta_class_rule, criterion):
    total_rule_loss = 0
    total_count = 0
    for meta_class in meta_class_rule:
        rule_ls = meta_class_rule[meta_class]
        # rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]

        for idx in range(len(rule_ls)):
            rule = rule_ls[idx]            
            # F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            curr_loss = do_apply_single_rule_per_sample(df, pred, meta_class, rule, criterion)
            # curr_loss = do_apply_single_rule_per_sample(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, criterion)
            total_rule_loss += curr_loss
            if curr_loss > 0:
                total_count += 1
    if total_count > 0:
        return total_rule_loss/total_count
    else:
        return total_rule_loss


def apply_rules_single_per_sample_evaluation(df, pred, meta_class_rule):
    total_rule_loss = 0
    total_count = 0
    total_violation_count = 0
    for meta_class in meta_class_rule:
        rule_ls = meta_class_rule[meta_class]
        # rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]

        for idx in range(len(rule_ls)):
            rule = rule_ls[idx]            
            # F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            violation_count = do_apply_single_rule_per_sample_eval(df, pred, meta_class, rule)
            # curr_loss = do_apply_single_rule_per_sample(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, criterion)
            total_violation_count += violation_count

    return total_violation_count

def apply_rules_single_per_sample_evaluation_full(df, pred, meta_class_rule):
    total_rule_loss = 0
    total_count = 0
    total_violation_count = 0
    total_violation_mappings = dict()
    for meta_class in meta_class_rule:
        rule_ls = meta_class_rule[meta_class]
        # rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]
        total_violation_mappings[meta_class] = dict()
        for idx in range(len(rule_ls)):
            rule = rule_ls[idx]            
            # F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            curr_violations = do_apply_single_rule_per_sample_eval_full(df, pred, meta_class, rule)
            # curr_loss = do_apply_single_rule_per_sample(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, criterion)
            # total_violation_count += violation_count
            total_violation_mappings[meta_class][idx] = curr_violations
    return total_violation_mappings


    # selected_df = selected_df[sample_id_boolean]

def evaluate_rule_violations(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings):
    total_rule_loss = 0
    total_rule_violations = 0
    for meta_class in meta_class_pred_boolean_mappings:
        pred_boolean_mappings_ls = meta_class_pred_boolean_mappings[meta_class]
        rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]

        for idx in range(len(pred_boolean_mappings_ls)):
            F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            curr_loss, curr_rule_violation = apply_single_rule_eval(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1)
            # logging.info("Loss::(%f, %f)", curr_loss, total_rule_loss)
            # logging.info("violations::(%d, %d)", curr_rule_violation, total_rule_violations)
            total_rule_loss += curr_loss
            total_rule_violations += curr_rule_violation

    return total_rule_loss, total_rule_violations


def obtain_rule_evaluations(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings):
    total_rule_loss = 0
    total_rule_violations = 0
    meta_class_pred_rule_label_mappings = dict()
    
    for meta_class in meta_class_pred_boolean_mappings:
        pred_boolean_mappings_ls = meta_class_pred_boolean_mappings[meta_class]
        rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]
        meta_class_pred_rule_label_mappings[meta_class] = []
        for idx in range(len(pred_boolean_mappings_ls)):
            F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            # df, pred, meta_class, pred_boolean_mappings, meta_class_mappings
            pred_rule_labels = obtain_rule_eval(df, pred, meta_class, pred_boolean_mappings_ls[idx], meta_class_mappings)
            meta_class_pred_rule_label_mappings[meta_class].append(pred_rule_labels.cpu().numpy())
            # curr_loss, curr_rule_violation = apply_single_rule_eval(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1)
            # logging.info("Loss::(%f, %f)", curr_loss, total_rule_loss)
            # logging.info("violations::(%d, %d)", curr_rule_violation, total_rule_violations)

    return meta_class_pred_rule_label_mappings

def post_eval_rule_f1_scores(sample_ids, meta_class_pred_labels_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, return_details=False):
    total_rule_loss = 0
    total_rule_violations = 0
    meta_class_f1_score_mappings = dict()
    for meta_class in meta_class_pred_boolean_mappings:
        pred_boolean_mappings_ls = meta_class_pred_boolean_mappings[meta_class]
        rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]
        pred_labels_ls = meta_class_pred_labels_mappings[meta_class]
        meta_class_f1_score_mappings[meta_class] = []
        for idx in range(len(pred_boolean_mappings_ls)):
            F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            # df, pred, meta_class, pred_boolean_mappings, meta_class_mappings
            # pred_rule_labels = obtain_rule_eval(df, pred, meta_class, pred_boolean_mappings_ls[idx], meta_class_mappings)
            
            rule_loss, rule_violations, F1_score = eval_rule_single(pred_labels_ls[idx][sample_ids], F1_score_low, F1_score_high)
            total_rule_loss += rule_loss
            total_rule_violations += rule_violations
            meta_class_f1_score_mappings[meta_class].append((F1_score, rule_violations))
            # curr_loss, curr_rule_violation = apply_single_rule_eval(df, pred, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings, sigmoidF1)
            # logging.info("Loss::(%f, %f)", curr_loss, total_rule_loss)
            # logging.info("violations::(%d, %d)", curr_rule_violation, total_rule_violations)
    if not return_details:
        return total_rule_loss, total_rule_violations
    else:
        return total_rule_loss, total_rule_violations, meta_class_f1_score_mappings

def reconstruct_rule_for_all(filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings):
    rule_scores_mappings = dict()
    for meta_class in filtered_meta_class_pred_boolean_mappings:
        for idx in range(len(filtered_meta_class_pred_boolean_mappings[meta_class])):
            rule = reconstruct_rule(meta_class, filtered_meta_class_pred_boolean_mappings[meta_class][idx])
            scores = filtered_meta_class_rule_score_mappings[meta_class][idx]
            rule_scores_mappings[rule] = list(scores)
            
    return rule_scores_mappings

def reconstruct_rule(meta_class, class_pred_boolean):
    rule = meta_class + " :- "
    count = 0
    for pred in class_pred_boolean:
        bool_val = class_pred_boolean[pred]
        if count >= 1:
            rule += ","
        if bool_val == 1:
            rule += pred
        else:
            rule += "!" + pred
        count += 1
    return rule

def construct_detailed_df_output(meta_class_f1_score_mappings, meta_class_pred_boolean_mappings):
    df_mappings = {"rule": [], "F1_score":[], "violation":[]}
    
    for meta_class in meta_class_pred_boolean_mappings:
        class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
        f1_score_tuple = meta_class_f1_score_mappings[meta_class]
        for idx in range(len(class_pred_boolean_ls)):
            rule = reconstruct_rule(meta_class, class_pred_boolean_ls[idx])
            df_mappings["rule"].append(rule)
            df_mappings["F1_score"].append(f1_score_tuple[idx][0])
            df_mappings["violation"].append(f1_score_tuple[idx][1])
    return df_mappings
        
def output_rule_predictions(all_meta_class_pred_rule_label_mappings, meta_class_pred_boolean_mappings):
    
    predict_df = dict()
    for meta_class in all_meta_class_pred_rule_label_mappings:
        class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
        for idx in range(len(all_meta_class_pred_rule_label_mappings[meta_class])):
            pred_rule_labels = all_meta_class_pred_rule_label_mappings[meta_class][idx]
            rule = reconstruct_rule(meta_class, class_pred_boolean_ls[idx])
            predict_df[rule] = pred_rule_labels
            
    return predict_df
       

def validate_rules(df, labels, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, mata_class_derived_rule_bounds_mappings):

    for meta_class in meta_class_pred_boolean_mappings:
        pred_boolean_mappings_ls = meta_class_pred_boolean_mappings[meta_class]
        rule_score_mappings_ls = meta_class_rule_score_mappings[meta_class]

        for idx in range(len(pred_boolean_mappings_ls)):
            F1_score_low, F1_score_high = rule_score_mappings_ls[idx][0], rule_score_mappings_ls[idx][1]
            curr_loss, curr_rule_violation, F1_score = validate_single_rule_eval(df, labels, meta_class, pred_boolean_mappings_ls[idx], F1_score_low, F1_score_high, meta_class_mappings)
            if F1_score >= 0:
                mata_class_derived_rule_bounds_mappings[meta_class][idx].append(F1_score)

# def check_consistency_rule_bound_mappings(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid):
#     filtered_meta_class_pred_boolean_mappings = dict()
#     filtered_meta_class_rule_score_mappings = dict()
    
#     for meta_class in mata_class_derived_rule_bound_mappings_training:
#         rule_bound_ls_training = mata_class_derived_rule_bound_mappings_training[meta_class]
#         rule_bound_ls_valid = mata_class_derived_rule_bound_mappings_valid[meta_class]
#         filtered_meta_class_pred_boolean_mappings[meta_class] = []
#         filtered_meta_class_rule_score_mappings[meta_class] = []
#         curr_meta_class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
#         for idx in range(len(rule_bound_ls_training)):
#             rule_bound_training = rule_bound_ls_training[idx]
#             rule_bound_valid = rule_bound_ls_valid[idx]
#             if rule_bound_valid[0] < 0 or rule_bound_valid[1] < 0 or rule_bound_training[0] < 0 or rule_bound_training[1] < 0:
#                 continue
#             min_lb = min(rule_bound_training[0], rule_bound_valid[0])
#             max_lb = max(rule_bound_training[0], rule_bound_valid[0])
#             min_hb = min(rule_bound_training[1], rule_bound_valid[1])
#             max_hb = max(rule_bound_training[1], rule_bound_valid[1])


#             if min_hb < max_lb:
#                 overlap = 0
#             elif max_hb == min_lb:
#                 overlap = 1
#             else:
#                 overlap = (min_hb - max_lb)/(max_hb - min_lb)

#             if overlap > 0.8:
#                 # print(rule_bound_training, rule_bound_valid)
#                 filtered_meta_class_pred_boolean_mappings[meta_class].append(curr_meta_class_pred_boolean_ls[idx])
#                 filtered_meta_class_rule_score_mappings[meta_class].append((rule_bound_training[0], rule_bound_training[1]))

#         print("number of rules for metaclass %s:%d"%(meta_class, len(filtered_meta_class_pred_boolean_mappings[meta_class])))

#     return filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings


def check_consistency_rule_bound_mappings(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid, topk=20):
    filtered_meta_class_pred_boolean_mappings = dict()
    filtered_meta_class_rule_score_mappings = dict()
    filtered_meta_class_rule_overlap_mappings = dict()
    
    for meta_class in mata_class_derived_rule_bound_mappings_training:
        rule_bound_ls_training = mata_class_derived_rule_bound_mappings_training[meta_class]
        rule_bound_ls_valid = mata_class_derived_rule_bound_mappings_valid[meta_class]
        filtered_meta_class_pred_boolean_mappings[meta_class] = []
        filtered_meta_class_rule_score_mappings[meta_class] = []
        filtered_meta_class_rule_overlap_mappings[meta_class] = []
        curr_meta_class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
        for idx in range(len(rule_bound_ls_training)):
            rule_bound_training = rule_bound_ls_training[idx]
            rule_bound_valid = rule_bound_ls_valid[idx]
            if rule_bound_valid[0] < 0 or rule_bound_valid[1] < 0 or rule_bound_training[0] < 0 or rule_bound_training[1] < 0:
                continue
            min_lb = min(rule_bound_training[0], rule_bound_valid[0])
            max_lb = max(rule_bound_training[0], rule_bound_valid[0])
            min_hb = min(rule_bound_training[1], rule_bound_valid[1])
            max_hb = max(rule_bound_training[1], rule_bound_valid[1])


            if min_hb < max_lb:
                overlap = 0
            elif max_hb == min_lb:
                overlap = 1
            else:
                overlap = (min_hb - max_lb)/(max_hb - min_lb)

            if overlap > 0.8 and rule_bound_training[1] - rule_bound_training[0] < 1 and rule_bound_training[1] - rule_bound_training[0] > 0:
            # if overlap > 0.8 and rule_bound_training[0] > 0:
                # print(rule_bound_training, rule_bound_valid)
                filtered_meta_class_pred_boolean_mappings[meta_class].append(curr_meta_class_pred_boolean_ls[idx])
                filtered_meta_class_rule_score_mappings[meta_class].append((rule_bound_training[0], rule_bound_training[1]))
                filtered_meta_class_rule_overlap_mappings[meta_class].append(overlap)
        overlap_score_ls = torch.tensor(filtered_meta_class_rule_overlap_mappings[meta_class])
        print("number of overlap rules for metaclass %s:%d"%(meta_class, len(overlap_score_ls)))
        sorted_overlap_sore_ls, sorted_ids = torch.sort(overlap_score_ls, descending=True)
        selected_sorted_ids = sorted_ids[0:topk].tolist()
        selected_pred_boolean_mappings = [filtered_meta_class_pred_boolean_mappings[meta_class][k] for k in selected_sorted_ids]
        filtered_meta_class_pred_boolean_mappings[meta_class] = selected_pred_boolean_mappings

        selected_rule_score_mappings = [filtered_meta_class_rule_score_mappings[meta_class][k] for k in selected_sorted_ids]
        filtered_meta_class_rule_score_mappings[meta_class] = selected_rule_score_mappings

        print("number of rules for metaclass %s:%d"%(meta_class, len(filtered_meta_class_pred_boolean_mappings[meta_class])))

    return filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings

def filter_rules_by_symbolic_conditions(pred_booleans):
    positive_count = 0
    negative_count = 0
    for pred in pred_booleans:
        if pred_booleans[pred] == 1:
            positive_count += 1
        else:
            negative_count += 1
    if positive_count == len(pred_booleans):
        return True
    
    else:
        if negative_count > 0 and positive_count >= 2:
            return True
        else:
            return False
                



def check_consistency_rule_bound_mappings_imagenet(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid, topk=20):
    filtered_meta_class_pred_boolean_mappings = dict()
    filtered_meta_class_rule_score_mappings = dict()
    filtered_meta_class_rule_overlap_mappings = dict()
    
    for meta_class in mata_class_derived_rule_bound_mappings_training:
        rule_bound_ls_training = mata_class_derived_rule_bound_mappings_training[meta_class]
        rule_bound_ls_valid = mata_class_derived_rule_bound_mappings_valid[meta_class]
        filtered_meta_class_pred_boolean_mappings[meta_class] = []
        filtered_meta_class_rule_score_mappings[meta_class] = []
        filtered_meta_class_rule_overlap_mappings[meta_class] = []
        curr_meta_class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
        for idx in range(len(rule_bound_ls_training)):
            rule_bound_training = rule_bound_ls_training[idx]
            curr_rule_pred_boolean = curr_meta_class_pred_boolean_ls[idx]
            keep_rule = filter_rules_by_symbolic_conditions(curr_rule_pred_boolean)
            if not keep_rule:
                continue
            
            rule_bound_valid = rule_bound_ls_valid[idx]
            if rule_bound_valid[0] < 0 or rule_bound_valid[1] < 0 or rule_bound_training[0] < 0 or rule_bound_training[1] < 0:
                continue
            min_lb = min(rule_bound_training[0], rule_bound_valid[0])
            max_lb = max(rule_bound_training[0], rule_bound_valid[0])
            min_hb = min(rule_bound_training[1], rule_bound_valid[1])
            max_hb = max(rule_bound_training[1], rule_bound_valid[1])


            if min_hb < max_lb:
                overlap = 0
            elif max_hb == min_lb:
                overlap = 1
            else:
                overlap = (min_hb - max_lb)/(max_hb - min_lb)

            if overlap > 0.6:# and rule_bound_training[1] - rule_bound_training[0] < 1 and rule_bound_training[1] - rule_bound_training[0] > 0:
            # if overlap > 0.8 and rule_bound_training[0] > 0:
                # print(rule_bound_training, rule_bound_valid)
                filtered_meta_class_pred_boolean_mappings[meta_class].append(curr_meta_class_pred_boolean_ls[idx])
                filtered_meta_class_rule_score_mappings[meta_class].append((rule_bound_training[0], rule_bound_training[1]))
                filtered_meta_class_rule_overlap_mappings[meta_class].append(overlap)
        overlap_score_ls = torch.tensor(filtered_meta_class_rule_overlap_mappings[meta_class])
        print("number of overlap rules for metaclass %s:%d"%(meta_class, len(overlap_score_ls)))
        sorted_overlap_sore_ls, sorted_ids = torch.sort(overlap_score_ls, descending=True)
        selected_sorted_ids = sorted_ids[0:topk].tolist()
        selected_pred_boolean_mappings = [filtered_meta_class_pred_boolean_mappings[meta_class][k] for k in selected_sorted_ids]
        filtered_meta_class_pred_boolean_mappings[meta_class] = selected_pred_boolean_mappings
        print(selected_pred_boolean_mappings)
        

        selected_rule_score_mappings = [filtered_meta_class_rule_score_mappings[meta_class][k] for k in selected_sorted_ids]
        print(selected_rule_score_mappings)
        filtered_meta_class_rule_score_mappings[meta_class] = selected_rule_score_mappings

        print("number of rules for metaclass %s:%d"%(meta_class, len(filtered_meta_class_pred_boolean_mappings[meta_class])))

    return filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings