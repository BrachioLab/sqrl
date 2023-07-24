from imagenet_x import load_annotations
import torch

from sklearn.metrics import f1_score

import numpy as np

from scipy.stats import bootstrap

from tqdm import tqdm

import pickle,os

annotations = load_annotations(partition="train")




cln_ls = list(annotations.columns[2:-3])

meta_class_ls = list(annotations["metaclass"].unique())

batch_size=256

rand_ids = torch.randperm(len(annotations))

train_count = int(len(rand_ids)*0.5)

train_ids = rand_ids[0:train_count]
valid_ids = rand_ids[train_count:]


train_annotation = annotations.iloc[train_ids.numpy()]
valid_annotation = annotations.iloc[valid_ids.numpy()]


def obtain_boolean_vals_quantiles(annotations):

    boolean_vals_by_class_by_clns = dict()

    f1_score_by_class_by_clns = dict()

    for _ in range(10):
        rand_ids = torch.randperm(len(annotations))
        
        for idx in tqdm(range(0, len(rand_ids), batch_size)):
            start_idx = idx
            end_idx = start_idx + batch_size
            if end_idx > len(rand_ids):
                end_idx = len(rand_ids)
                
            curr_rand_ids = rand_ids[start_idx:end_idx]
            
            curr_rand_annotations = annotations.iloc[curr_rand_ids.numpy()]
            
            
            for meta_class in meta_class_ls:
                if meta_class not in boolean_vals_by_class_by_clns:
                    boolean_vals_by_class_by_clns[meta_class] = dict()                
                for cln in cln_ls:
                    if cln not in boolean_vals_by_class_by_clns[meta_class]:
                        boolean_vals_by_class_by_clns[meta_class][cln] = []
                    
                    curr_subset = curr_rand_annotations[curr_rand_annotations[cln] == 1]
                    if len(curr_subset) <= 0:
                        continue
                    
                    class_labels =  np.array(curr_subset["metaclass"] == meta_class).astype(int)
                    class_ones = np.ones(len(class_labels))
                    curr_f1_score = f1_score(class_ones, class_labels)
                    boolean_vals_by_class_by_clns[meta_class][cln].append(curr_f1_score)
                    
    
    for meta_class in meta_class_ls:
        f1_score_by_class_by_clns[meta_class] = dict()
        for cln in cln_ls:
            stat_array = np.array(boolean_vals_by_class_by_clns[meta_class][cln])
            if len(stat_array) <= 0:
                f1_score_by_class_by_clns[meta_class][cln] = []
                continue

            f1_low_low, f1_low_high = bootstrap([stat_array, ], np.min, confidence_level=0.99, method='percentile').confidence_interval

            f1_high_low, f1_high_high = bootstrap([stat_array, ], np.max, confidence_level=0.99, method='percentile').confidence_interval


            f1_score_by_class_by_clns[meta_class][cln] = [f1_low_low, f1_high_high]
            
            
    return f1_score_by_class_by_clns
    # return boolean_vals_by_class_by_clns
            

train_f1_score_by_class_by_clns = obtain_boolean_vals_quantiles(train_annotation)
valid_f1_score_by_class_by_clns = obtain_boolean_vals_quantiles(valid_annotation)

cache_path = "/data6/wuyinjun/imagenet/"

with open(os.path.join(cache_path, "train_f1_scores"), "wb") as f:
    pickle.dump(train_f1_score_by_class_by_clns, f)


with open(os.path.join(cache_path, "valid_f1_scores"), "wb") as f:
    pickle.dump(valid_f1_score_by_class_by_clns, f)

# cache_path = "/data6/wuyinjun/imagenet/"

# with open(os.path.join(cache_path, "train_f1_scores")) as f:
#     train_f1_score_by_class_by_clns = pickle.load(f)


# with open(os.path.join(cache_path, "valid_f1_scores")) as f:
#     valid_f1_score_by_class_by_clns = pickle.load(f)

filtered_rules = dict()
filtered_rule_f1_scores = dict()

for meta_class in meta_class_ls:
    if meta_class not in filtered_rules:
        filtered_rules[meta_class] = []
        filtered_rule_f1_scores[meta_class] = []
    
    
    for cln in cln_ls:
        if len(train_f1_score_by_class_by_clns[meta_class][cln]) <= 0:
            continue
        if len(valid_f1_score_by_class_by_clns[meta_class][cln]) <= 0:
            continue

        
        train_f1_score_min, train_f1_score_max = train_f1_score_by_class_by_clns[meta_class][cln]
        valid_f1_score_min, valid_f1_score_max = valid_f1_score_by_class_by_clns[meta_class][cln]
        

        min_lb = min(train_f1_score_min, valid_f1_score_min)
        max_lb = max(train_f1_score_min, valid_f1_score_min)
        min_hb = min(train_f1_score_max, valid_f1_score_max)
        max_hb = max(train_f1_score_max, valid_f1_score_max)


        if min_hb < max_lb:
            overlap = 0
        elif max_hb == min_lb:
            overlap = 1
        else:
            overlap = (min_hb - max_lb)/(max_hb - min_lb)
        if overlap > 0.8:
            filtered_rules[meta_class].append({cln:1})
            filtered_rule_f1_scores[meta_class].append([min_lb, max_hb])
        
        print(overlap)

with open(os.path.join(cache_path, "filtered_rules"), "wb") as f:
    pickle.dump(filtered_rules, f)


with open(os.path.join(cache_path, "filtered_rule_f1_scores"), "wb") as f:
    pickle.dump(filtered_rule_f1_scores, f)


print(annotations)

annotations["class"]

annotations["metaclass"]