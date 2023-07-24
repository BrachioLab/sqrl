import argparse
import os
from util import *

from parse_args import *

import numpy as np

from scipy.stats import skew
from tqdm import tqdm

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

tp_range = {
    "all": (-np.inf, np.inf),
    "first4days": (0, 0, 96),
    "first8days": (0, 0, 8 * 24),
    "last12hours": (1, -12, -1),
    "first25percent": (2, 0, 0.25),
    "first50percent": (2, 0, 0.5)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
               (3, 10), (3, 25), (3, 50)]


def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.stack(extracted_features, axis=0)


def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])

def extract_features2(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])



def convert_to_dict(data):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        # channel = header[i]
        # if len(channel_info[channel]['possible_values']) != 0:
        #     ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret


def conver_df_to_mat(args, feature_array_ls):
    data = [convert_to_dict(X) for X in feature_array_ls]
    return extract_features(data, args.period, args.features)

def convert_df_to_mat_with_time_steps(args, feature_array_ls):
    
    period_map = tp_range[args.period]
    
    all_df_ls = []
    all_tp_ls = []
    all_mask_ls = []
    
    
    for feature_arr in tqdm(feature_array_ls):
        feature_arr = feature_arr.astype(np.float32)
        tp_ls = feature_arr[:,0]
        if period_map[0] == 0:
            lb = period_map[1]
            ub = period_map[2]
            selected_df = feature_arr[(tp_ls >= lb) &  (tp_ls < ub)][:,1:]
            selected_tp = tp_ls[(tp_ls >= lb) &  (tp_ls < ub)]
        elif period_map[0] == 1:
            back_lb = np.max(tp_ls) + period_map[1]
            selected_df = feature_arr[(tp_ls >= back_lb)][:,1:]
            selected_tp = tp_ls[(tp_ls >= back_lb)]
        
        # selected_tp_ls = np.array(selected_df.index)
        # selected_df_ls = np.array(selected_df)
        
        selected_mask_ls = 1 - np.isnan(selected_df)
        selected_df[np.isnan(selected_df)] = 0
        
        all_df_ls.append(selected_df)
        all_tp_ls.append(selected_tp)
        all_mask_ls.append(selected_mask_ls)
    
    return all_df_ls, all_tp_ls, all_mask_ls
    # data = [convert_to_dict(X) for X in feature_array_ls]
    
    
    
    # return extract_features(data, args.period, args.features)


def load_and_process_data(args, task_output_folder_name, prefix):
    if args.imputed:
        df_file_name = os.path.join(task_output_folder_name, "df_imputed_" + prefix)
        
    else:
        df_file_name = os.path.join(task_output_folder_name, "all_" + prefix + "_df_ls")

    feature_files = load_objs(df_file_name)

    feature_array_ls = [np.array(feature_df.reset_index()) for feature_df in feature_files]

    all_df_ls, all_tp_ls, all_mask_ls = convert_df_to_mat_with_time_steps(args, feature_array_ls)

    label_file_name = os.path.join(task_output_folder_name, "all_" + prefix + "_label_ls")

    labels = load_objs(label_file_name)

    labels_mat = np.array(labels)

    if args.imputed:
        feature_file = os.path.join(task_output_folder_name, prefix + "_X_imputed")
        mask_file = os.path.join(task_output_folder_name, prefix + "_mask_imputed")
        tp_file = os.path.join(task_output_folder_name, prefix + "_tp_imputed")
    else:
        feature_file = os.path.join(task_output_folder_name, prefix + "_X")
        mask_file = os.path.join(task_output_folder_name, prefix + "_mask")
        tp_file = os.path.join(task_output_folder_name, prefix + "_tp")

    save_objs(all_df_ls, feature_file)
    save_objs(all_tp_ls, tp_file)
    save_objs(all_mask_ls, mask_file)

    if args.imputed:
        label_file = os.path.join(task_output_folder_name, prefix + "_Y_imputed")
    else:
        label_file = os.path.join(task_output_folder_name, prefix + "_Y")

    save_objs(labels_mat, label_file)

def main(args):
    task_output_folder_name = os.path.join(args.output, args.task_name)

    load_and_process_data(args, task_output_folder_name, "train")

    load_and_process_data(args, task_output_folder_name, "val")

    load_and_process_data(args, task_output_folder_name, "test")


if __name__ == '__main__':

    args = parse_args()
    main(args)
