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

def calculate_statistic_per_attribute_value(df, attr1, attr2):
    groupby_res = df.groupby(attr1)[attr2].apply(list)
    groupby_keys = list(groupby_res.keys())
    estimated_gmm_by_keys = {}
    for key in groupby_keys:
        agg_ls = np.array(groupby_res[key])
        print("attribute type::", key, agg_ls.dtype)    
        if test_numeric(agg_ls):
            estimated_gmm_density = estimate_gmm(agg_ls)
        else:
            estimated_gmm_density =  estimate_discrete(agg_ls)#estimate_gmm(agg_ls)
        estimated_gmm_by_keys[key] = estimated_gmm_density

    return estimated_gmm_by_keys




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


def retrieve_value_pair_ls_for_correlated_attribute_main(args, coorelated_attr_pair_ls):
    all_estimated_gmm_by_keys = {}
    all_coorelated_attrs = [list(coorelated_attr_pair_ls[k])[0] for k in range(len(coorelated_attr_pair_ls))] + [list(coorelated_attr_pair_ls[k])[1] for k in range(len(coorelated_attr_pair_ls))]

    all_coorelated_attrs = set(all_coorelated_attrs)

    prefix = "train"

    print("retrieve training dataframe")
    all_correlated_values_in_df_ls = retrieve_df_label_ls(args, prefix, all_coorelated_attrs)
    print("retrieve validation dataframe")
    retrieve_df_label_ls(args, "val", all_coorelated_attrs)
    print("retrieve test dataframe")
    retrieve_df_label_ls(args, "test", all_coorelated_attrs)
        # all_correlated_values_in_df.to_csv(all_df_file_name)
    # else:
    #     all_correlated_values_in_df_ls = load_objs(all_df_file_name)
    #     all_correlated_values_in_df = pd.read_csv(all_df_file_name)

    print("start estimating distributions")
    for coorelated_attrs in coorelated_attr_pair_ls:
        attr1, attr2 = list(coorelated_attrs)[0], list(coorelated_attrs)[1]
        print("processing attributes::", attr1, attr2)
        correlated_values_in_df = retrieve_correlated_values_by_attributes(all_correlated_values_in_df_ls, coorelated_attrs)
        if correlated_values_in_df is None:
            continue
        
        all_estimated_gmm_by_keys[attr_ls_to_str([attr1, attr2])] = calculate_statistic_per_attribute_value(correlated_values_in_df, attr1, attr2)
        all_estimated_gmm_by_keys[attr_ls_to_str([attr2, attr1])] = calculate_statistic_per_attribute_value(correlated_values_in_df, attr2, attr1)
        # print()

    task_output_folder = os.path.join(args.output, args.task_name)

    all_estimated_dist_file = os.path.join(task_output_folder, "all_dist_mappings")

    save_objs(all_estimated_gmm_by_keys, all_estimated_dist_file)

    return all_estimated_gmm_by_keys, all_correlated_values_in_df_ls


def main(args):
    correlated_attr_pair_file_name = os.path.join(args.output, "correlated_attrs")
    coorelated_attr_pair_ls = read_correlated_attributes(correlated_attr_pair_file_name)
    all_estimated_gmm_by_keys, all_correlated_values_in_df_ls = retrieve_value_pair_ls_for_correlated_attribute_main(args, coorelated_attr_pair_ls)

    

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
