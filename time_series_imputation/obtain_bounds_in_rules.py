import os, sys

import argparse
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from check_correlations.correlation import *
from sklearn import mixture
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from impute_main import *

from util import *

from discrete_dist import *

import data_reader

from parse_args import *

from scipy.stats import bootstrap

from physionet_dataset import get_dataloader

from physionet_dataset import attributes as phy_attributes

from scipy.stats import norm

def estimate_gmm(val_arr):
    # lowest_bic = np.infty
    # bic = []
    # for n_components in range(10):
    #     gmm = mixture.GaussianMixture(
    #         n_components=n_components, covariance_type="full"
    #     )
    #     gmm.fit(val_arr)
    #     bic.append(gmm.bic(val_arr))
    #     if bic[-1] < lowest_bic:
    #         lowest_bic = bic[-1]
    #         best_gmm = gmm

    kde = KernelDensity(kernel='gaussian', bandwidth=0.005).fit(val_arr.reshape(-1,1))

    return kde

def estimate_discrete(val_arr):
    dist =  discrete_dist(val_arr)
    return dist



    # return np.all(np.core.defchararray.isnumeric(array))

def calculate_statistic_per_attribute_value(df, attr1, attr2, df_valid=None):
    groupby_res = df.groupby(attr1)[attr2].apply(list)    
    if df_valid is not None:
        groupby_res_valid = df_valid.groupby(attr1)[attr2].apply(list)
        groupby_keys_valid = list(groupby_res_valid.keys())
    groupby_keys = list(groupby_res.keys())
    estimated_gmm_by_keys = {}
    for key in groupby_keys:
        if df_valid is not None:
            if key not in groupby_keys_valid:
                continue
        
        agg_ls = np.array(groupby_res[key])
        agg_ls_valid = np.array(groupby_res_valid[key])

        
        if len(agg_ls) < 200 or len(agg_ls_valid) < 200:
            continue
        
        
        common_val_set = set(agg_ls).intersection(set(agg_ls_valid))
        if len(common_val_set) <= 0:
            continue
        
        print("attribute type::", key, agg_ls.dtype)    
        agg_ls = [item for item in agg_ls if item in common_val_set]
        agg_ls = np.array(agg_ls)
        if test_numeric(agg_ls):
            estimated_gmm_density = estimate_gmm(agg_ls)
        else:
            estimated_gmm_density =  estimate_discrete(agg_ls)#estimate_gmm(agg_ls)
        estimated_gmm_by_keys[key] = estimated_gmm_density

    return estimated_gmm_by_keys



def calculate_confidence_interval(stat_ls):
    stat_array = np.array(stat_ls)
    stat_array = stat_array.reshape(-1)
    rng = np.random.default_rng()

    f1_low_low, f1_low_high = bootstrap([stat_array, ], np.min, confidence_level=0.95, method='percentile', n_resamples=100).confidence_interval

    f1_high_low, f1_high_high = bootstrap([stat_array, ], np.max, confidence_level=0.95, method='percentile', n_resamples=100).confidence_interval


    return f1_low_low, f1_low_high, f1_high_low, f1_high_high

def calculate_statistic_single_attribute_value(df, attr1, df_valid=None):
    df_attr_ls = df[attr1]#[attr2].apply(list)    
    # if df_valid is not None:
    #     groupby_res_valid = df_valid.groupby(attr1)[attr2].apply(list)
    #     groupby_keys_valid = list(groupby_res_valid.keys())
    df_attr_ls = np.array(df_attr_ls)
    
    f1_low_low, f1_low_high, f1_high_low, f1_high_high = calculate_confidence_interval(df_attr_ls)
    
    df_valid_attr_ls = df_valid[attr1]
    df_valid_attr_ls = np.array(df_valid_attr_ls)
    
    f1_low_low_valid, f1_low_high_valid, f1_high_low_valid, f1_high_high_valid = calculate_confidence_interval(df_valid_attr_ls)
    
    interval = [max(f1_low_low_valid, f1_low_low), min(f1_high_high, f1_high_high_valid)]
    
    
    
    # estimated_gmm_by_keys = {}
    # for key in groupby_keys:
    #     if df_valid is not None:
    #         if key not in groupby_keys_valid:
    #             continue
        
    #     agg_ls = np.array(groupby_res[key])
    #     agg_ls_valid = np.array(groupby_res_valid[key])

        
    #     if len(agg_ls) < 200 or len(agg_ls_valid) < 200:
    #         continue
        
        
    #     common_val_set = set(agg_ls).intersection(set(agg_ls_valid))
    #     if len(common_val_set) <= 0:
    #         continue
        
    #     print("attribute type::", key, agg_ls.dtype)    
    #     agg_ls = [item for item in agg_ls if item in common_val_set]
    #     agg_ls = np.array(agg_ls)
    #     if test_numeric(agg_ls):
    #         estimated_gmm_density = estimate_gmm(agg_ls)
    #     else:
    #         estimated_gmm_density =  estimate_discrete(agg_ls)#estimate_gmm(agg_ls)
    #     estimated_gmm_by_keys[key] = estimated_gmm_density

    return interval




def retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls, coorelated_attrs):
    all_non_empty_df = None

    for k in tqdm(range(len(all_correlated_values_in_df_ls))):
        non_empty_df = all_correlated_values_in_df_ls[k][list(coorelated_attrs)].dropna()
        
        if len(non_empty_df) > 0:
            if all_non_empty_df is None:
                all_non_empty_df = non_empty_df
            else:
                all_non_empty_df = pd.concat([all_non_empty_df, non_empty_df], ignore_index=True)

    return all_non_empty_df

# def impute_records_for_each_row():
def retrieve_label_mappings(args, prefix):
    label_file_name = os.path.join(args.input, args.task_name, prefix + "_listfile.csv")

    curr_prefix = args.task_name.replace("-", "_")

    read_label_file_func_name = "read_" + curr_prefix + "_label_file"

    read_label_file_func = getattr(data_reader, read_label_file_func_name)

    label_mappings = read_label_file_func(label_file_name)

    return label_mappings


def retrieve_df_label_ls(args, prefix, all_coorelated_attrs):
    task_output_folder = os.path.join(args.output, args.task_name)

    if not os.path.exists(task_output_folder):
        os.makedirs(task_output_folder)

    all_df_file_name = os.path.join(task_output_folder, "all_" + prefix + "_df_ls")
    all_label_file_name = os.path.join(task_output_folder, "all_" + prefix + "_label_ls")
    # if not os.path.exists(all_df_file_name):
    if prefix == "val":
        input_train_folder = os.path.join(args.input, args.task_name, "train/")    
    else:
        input_train_folder = os.path.join(args.input, args.task_name, prefix + "/")

    train_label_mappings = retrieve_label_mappings(args, prefix)

    all_correlated_values_in_df_ls, all_train_label_ls = retrieve_value_pairs_for_correlated_attrs(input_train_folder, train_label_mappings, list(all_coorelated_attrs))
    # all_correlated_values_in_df_ls = retrieve_value_pairs_for_correlated_attrs(list(all_coorelated_attrs))
    save_objs(all_correlated_values_in_df_ls, all_df_file_name)
    save_objs(all_train_label_ls, all_label_file_name)
    return all_correlated_values_in_df_ls


def retrieve_df_label_ls_single_attr(args, prefix):
    task_output_folder = os.path.join(args.output, args.task_name)

    if not os.path.exists(task_output_folder):
        os.makedirs(task_output_folder)

    all_df_file_name = os.path.join(task_output_folder, "all_" + prefix + "_df_ls")
    all_label_file_name = os.path.join(task_output_folder, "all_" + prefix + "_label_ls")
    # if not os.path.exists(all_df_file_name):
    if prefix == "val":
        input_train_folder = os.path.join(args.input, args.task_name, "train/")    
    else:
        input_train_folder = os.path.join(args.input, args.task_name, prefix + "/")

    train_label_mappings = retrieve_label_mappings(args, prefix)

    all_correlated_values_in_df_ls, all_train_label_ls = retrieve_value_pairs_for_single_attrs(input_train_folder, train_label_mappings)
    # all_correlated_values_in_df_ls = retrieve_value_pairs_for_correlated_attrs(list(all_coorelated_attrs))
    save_objs(all_correlated_values_in_df_ls, all_df_file_name)
    save_objs(all_train_label_ls, all_label_file_name)
    return all_correlated_values_in_df_ls


def retrieve_value_pair_ls_for_correlated_attribute_main(args, coorelated_attr_pair_ls):
    all_estimated_gmm_by_keys = {}
    all_coorelated_attrs = [list(coorelated_attr_pair_ls[k])[0] for k in range(len(coorelated_attr_pair_ls))] + [list(coorelated_attr_pair_ls[k])[1] for k in range(len(coorelated_attr_pair_ls))]

    all_coorelated_attrs = set(all_coorelated_attrs)

    prefix = "train"

    print("retrieve training dataframe")
    if os.path.exists(os.path.join(args.output, "train_df_cor_values_ls")):
        all_correlated_values_in_df_ls = load_objs(os.path.join(args.output, "train_df_cor_values_ls"))
    else:
        all_correlated_values_in_df_ls = retrieve_df_label_ls(args, prefix, all_coorelated_attrs)
        save_objs(all_correlated_values_in_df_ls, os.path.join(args.output, "train_df_cor_values_ls"))
    
    print("retrieve validation dataframe")
    if os.path.exists(os.path.join(args.output, "valid_df_cor_values_ls")):
        all_correlated_values_in_df_ls_valid = load_objs(os.path.join(args.output, "valid_df_cor_values_ls"))
    else:
        all_correlated_values_in_df_ls_valid = retrieve_df_label_ls(args, "val", all_coorelated_attrs)
        save_objs(all_correlated_values_in_df_ls_valid, os.path.join(args.output, "valid_df_cor_values_ls"))
    print("retrieve test dataframe")
    if os.path.exists(os.path.join(args.output, "test_df_cor_values_ls")):
        all_correlated_values_in_df_ls_test = load_objs(os.path.join(args.output, "test_df_cor_values_ls"))
    else:
        all_correlated_values_in_df_ls_test = retrieve_df_label_ls(args, "test", all_coorelated_attrs)
        save_objs(all_correlated_values_in_df_ls_test, os.path.join(args.output, "test_df_cor_values_ls"))
        # all_correlated_values_in_df.to_csv(all_df_file_name)
    # else:
    #     all_correlated_values_in_df_ls = load_objs(all_df_file_name)
    #     all_correlated_values_in_df = pd.read_csv(all_df_file_name)

    print("start estimating distributions")
    for coorelated_attrs in coorelated_attr_pair_ls:
        attr1, attr2 = list(coorelated_attrs)[0], list(coorelated_attrs)[1]
        print("processing attributes::", attr1, attr2)
        correlated_values_in_df = retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls, coorelated_attrs)
        correlated_values_in_df_valid = retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls_valid, coorelated_attrs)
        if correlated_values_in_df is None:
            continue
        if correlated_values_in_df_valid is not None and len(correlated_values_in_df_valid) > 0:
        
            all_estimated_gmm_by_keys[attr_ls_to_str([attr1, attr2])] = calculate_statistic_per_attribute_value(correlated_values_in_df, attr1, attr2, correlated_values_in_df_valid)
            all_estimated_gmm_by_keys[attr_ls_to_str([attr2, attr1])] = calculate_statistic_per_attribute_value(correlated_values_in_df, attr2, attr1, correlated_values_in_df_valid)
        # print()

    task_output_folder = os.path.join(args.output, args.task_name)

    all_estimated_dist_file = os.path.join(task_output_folder, "all_dist_mappings")

    save_objs(all_estimated_gmm_by_keys, all_estimated_dist_file)

    return all_estimated_gmm_by_keys, all_correlated_values_in_df_ls



def retrieve_value_pair_ls_for_single_attribute_main(args):
    all_estimated_attr_range_mappings = {}
    # all_coorelated_attrs = [list(coorelated_attr_pair_ls[k])[0] for k in range(len(coorelated_attr_pair_ls))] + [list(coorelated_attr_pair_ls[k])[1] for k in range(len(coorelated_attr_pair_ls))]

    # all_coorelated_attrs = set(all_coorelated_attrs)

    prefix = "train"

    print("retrieve training dataframe")
    cache_train_df_name = os.path.join(args.output, "train_df_single_values_ls")
    cache_valid_df_name = os.path.join(args.output, "valid_df_single_values_ls")
    cache_test_df_name = os.path.join(args.output, "test_df_single_values_ls")
    if os.path.exists(cache_train_df_name):
        all_correlated_values_in_df_ls = load_objs(cache_train_df_name)
    else:
        all_correlated_values_in_df_ls = retrieve_df_label_ls_single_attr(args, prefix)
        save_objs(all_correlated_values_in_df_ls, cache_train_df_name)
    
    print("retrieve validation dataframe")
    if os.path.exists(cache_valid_df_name):
        all_correlated_values_in_df_ls_valid = load_objs(cache_valid_df_name)
    else:
        all_correlated_values_in_df_ls_valid = retrieve_df_label_ls_single_attr(args, "val")
        save_objs(all_correlated_values_in_df_ls_valid, cache_valid_df_name)
    print("retrieve test dataframe")
    if os.path.exists(cache_test_df_name):
        all_correlated_values_in_df_ls_test = load_objs(cache_test_df_name)
    else:
        all_correlated_values_in_df_ls_test = retrieve_df_label_ls_single_attr(args, "test")
        save_objs(all_correlated_values_in_df_ls_test, cache_test_df_name)
        # all_correlated_values_in_df.to_csv(all_df_file_name)
    # else:
    #     all_correlated_values_in_df_ls = load_objs(all_df_file_name)
    #     all_correlated_values_in_df = pd.read_csv(all_df_file_name)
    cln_names = list(all_correlated_values_in_df_ls[0].columns)
    cln_name_ls = [[cln_names[k]] for k in range(len(cln_names))]
    print("start estimating distributions")
    for coorelated_attrs in cln_name_ls:
        # attr1, attr2 = list(coorelated_attrs)[0], list(coorelated_attrs)[1]
        print("processing attributes::", coorelated_attrs)
        correlated_values_in_df = retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls, coorelated_attrs)
        correlated_values_in_df_valid = retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls_valid, coorelated_attrs)
        if correlated_values_in_df is None:
            continue
        if correlated_values_in_df_valid is not None and len(correlated_values_in_df_valid) > 0:
        
            all_estimated_attr_range_mappings[coorelated_attrs[0]] = calculate_statistic_single_attribute_value(correlated_values_in_df, coorelated_attrs, correlated_values_in_df_valid)
            all_estimated_attr_range_mappings[coorelated_attrs[0]] = calculate_statistic_single_attribute_value(correlated_values_in_df, coorelated_attrs, correlated_values_in_df_valid)
        # print()

    task_output_folder = os.path.join(args.output, args.task_name)

    all_estimated_dist_file = os.path.join(task_output_folder, "all_attr_range_mappings")

    save_objs(all_estimated_attr_range_mappings, all_estimated_dist_file)

    return all_estimated_attr_range_mappings, all_correlated_values_in_df_ls

def obtain_estimated_attr_ranges(train_dataset, valid_dataset):
    all_estimated_attr_range_mappings = []
    orig_feat_values = train_dataset.origin_observed_values
    observed_masks = train_dataset.observed_masks
    valid_orig_feat_values = valid_dataset.origin_observed_values
    valid_observed_masks = valid_dataset.observed_masks
    
    for idx in range(orig_feat_values.shape[-1]):
        df_attr_ls = orig_feat_values[:,:,idx]
        curr_mask = observed_masks[:,:,idx]
        df_attr_ls = df_attr_ls[curr_mask == 1]
        
        df_valid_attr_ls = valid_orig_feat_values[:,:,idx]
        curr_mask = valid_observed_masks[:,:,idx]
        df_valid_attr_ls = df_valid_attr_ls[curr_mask == 1]
        
        
        f1_low_low, f1_low_high, f1_high_low, f1_high_high = calculate_confidence_interval(df_attr_ls)
    
        mu, std = norm.fit(df_attr_ls.numpy())
        
        f1_low_low_valid, f1_low_high_valid, f1_high_low_valid, f1_high_high_valid = calculate_confidence_interval(df_valid_attr_ls)
    

    
        interval = [max(f1_low_low_valid, f1_low_low), min(f1_high_high, f1_high_high_valid)]
        
        print("stats on attribute ", phy_attributes[idx])
        
        print(interval)
        
        # print("mu:", mu)
        
        # print("std:", std)
        
        all_estimated_attr_range_mappings.append(interval)
    return all_estimated_attr_range_mappings

def main(args):
    # correlated_attr_pair_file_name = os.path.join(args.output, "correlated_attrs")
    # coorelated_attr_pair_ls = read_correlated_attributes(correlated_attr_pair_file_name)
    # if args.dataset == 'mimic3':
    #     all_estimated_gmm_by_keys, all_correlated_values_in_df_ls = retrieve_value_pair_ls_for_single_attribute_main(args)
    # elif args.dataset == "physionet":
    train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader, data_max, data_min = get_dataloader(root_data_path=args.input)
    all_estimated_attr_range_mappings = obtain_estimated_attr_ranges(train_dataset, valid_dataset)

    all_estimated_dist_file = os.path.join(args.output, "all_attr_range_mappings")

    save_objs(all_estimated_attr_range_mappings, all_estimated_dist_file)
if __name__ == '__main__':


    # mb_size = 128
    # # 2. Missing rate
    # p_miss = 0.2
    # # 3. Hint rate
    # p_hint = 0.9
    # # 4. Loss Hyperparameters
    # alpha = 10
    # # 5. Train Rate
    # train_rate = 0.8

    args = parse_args()

    main(args)
