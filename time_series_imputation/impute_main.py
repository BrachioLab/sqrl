import numpy as np
from util import *
import argparse
import os
from discrete_dist import *
from tqdm import tqdm
from parse_args import *
import numpy as np

tol = 0.1

def impute_values_by_nearest_key(estimated_gmm_by_keys, unique_val):
    key_ls = np.array(list(estimated_gmm_by_keys.keys()))
    if not test_numeric(key_ls):
        if unique_val not in estimated_gmm_by_keys:
            return None
        else:
            return estimated_gmm_by_keys[unique_val]
    
    else:
        min_dist = np.infty
        min_key = None

        for key in key_ls:
            curr_dist = np.abs(unique_val - key)
            if curr_dist < min_dist:
                min_key = key
                min_dist = curr_dist

        if min_dist < tol:
            return estimated_gmm_by_keys[min_key]
        else:
            return None




def fill_missing_vals_by_attrs(df_copy, df_not_na_flag, attr1, attr2, all_estimated_gmm_by_keys):
    selected_row_ids = (df_not_na_flag[attr1] == True) & (df_not_na_flag[attr2] == False)

    if np.sum(np.array(selected_row_ids)) <= 0:
        return

    attr1_unique_vals = df_copy.loc[selected_row_ids,attr1].unique()

    key = attr_ls_to_str([attr1, attr2])

    if not key in all_estimated_gmm_by_keys:
        return 

    estimated_gmm_by_keys = all_estimated_gmm_by_keys[key]

    pre_missing_ratio = np.sum(np.array(df_copy[[attr1, attr2]].isnull()))*1.0/(len(df_copy)*2)

    pre_non_empty_count = np.sum(np.array(df_copy[[attr1, attr2]].isnull()))

    # print("missing ratio before imputing::", pre_missing_ratio)

    for unique_val in attr1_unique_vals:
        dist = impute_values_by_nearest_key(estimated_gmm_by_keys, unique_val)
        if dist is None:
            continue
        # if unique_val not in estimated_gmm_by_keys:
        #     continue
        # dist = estimated_gmm_by_keys[unique_val]


        filter_condition = (df_copy[attr1] == unique_val) & (df_not_na_flag[attr2] == False)

        sampled_count = np.sum(np.array(filter_condition))

        if sampled_count > 0:
            sampled_vals = dist.sample(sampled_count)

            df_copy.loc[filter_condition, attr2] = sampled_vals.reshape(-1)


    post_missing_ratio = np.sum(np.array(df_copy[[attr1, attr2]].isnull()))*1.0/(len(df_copy)*2)
    post_non_empty_count = np.sum(np.array(df_copy[[attr1, attr2]].isnull()))

    # print("missing ratio after imputing::", post_missing_ratio)
    # print()

def impute_records_main(df, coorelated_attr_pair_ls, all_estimated_gmm_by_keys):
    df_not_na_flag = df.notna()
    df_copy = df.copy()

    pre_missing_ratio = np.sum(np.array(df_copy.isnull()))*1.0/(len(df_copy)*len(df_copy.columns))

    pre_non_empty_count = np.sum(np.array(df_copy.isnull()))

    print("missing ratio before imputing::", pre_missing_ratio)

    for coorelated_attrs in coorelated_attr_pair_ls:
        attr1, attr2 = list(coorelated_attrs)[0], list(coorelated_attrs)[1]
        
        fill_missing_vals_by_attrs(df_copy, df_not_na_flag, attr1, attr2, all_estimated_gmm_by_keys)        
        fill_missing_vals_by_attrs(df_copy, df_not_na_flag, attr2, attr1, all_estimated_gmm_by_keys)        

    post_missing_ratio = np.sum(np.array(df_copy.isnull()))*1.0/(len(df_copy)*len(df_copy.columns))
    post_non_empty_count = np.sum(np.array(df_copy.isnull()))

    print("missing ratio after imputing::", post_missing_ratio)

    
    return df_copy, pre_missing_ratio, post_missing_ratio

def impute_main(prefix, task_output_folder_name, coorelated_attr_pair_ls, all_estimated_gmm_by_keys):
    all_df_file_name = os.path.join(task_output_folder_name, "all_" + prefix + "_df_ls")

    all_correlated_values_in_df_ls = load_objs(all_df_file_name)

    df_imputed_ls = []
    df_missing_ratio_ls = []
    for df in tqdm(all_correlated_values_in_df_ls):
        df_imputed, pre_missing_ratio, post_missing_ratio = impute_records_main(df, coorelated_attr_pair_ls, all_estimated_gmm_by_keys)
        df_imputed_ls.append(df_imputed)
        df_missing_ratio_ls.append((pre_missing_ratio, post_missing_ratio))

    df_imputed_file_name = os.path.join(task_output_folder_name, "df_imputed_" + prefix)

    df_missing_ratio_file_name = os.path.join(task_output_folder_name, "df_imputed_missing_ratio_" + prefix)

    save_objs(df_imputed_ls, df_imputed_file_name)
    save_objs(df_missing_ratio_ls, df_missing_ratio_file_name)

def main(args):
    task_output_folder_name = os.path.join(args.output, args.task_name)

    if not os.path.exists(task_output_folder_name):
        os.makedirs(task_output_folder_name)

    correlated_attr_pair_file_name = os.path.join(args.output, "correlated_attrs")
    coorelated_attr_pair_ls = read_correlated_attributes(correlated_attr_pair_file_name)
    all_estimated_dist_file = os.path.join(task_output_folder_name, "all_dist_mappings")

    all_estimated_gmm_by_keys = load_objs(all_estimated_dist_file)


    impute_main("train", task_output_folder_name, coorelated_attr_pair_ls, all_estimated_gmm_by_keys)
    impute_main("val", task_output_folder_name, coorelated_attr_pair_ls, all_estimated_gmm_by_keys)
    impute_main("test", task_output_folder_name, coorelated_attr_pair_ls, all_estimated_gmm_by_keys)

    

    

if __name__ == '__main__':


    args = parse_args()

    main(args)

