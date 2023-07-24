import os
import argparse

import pandas as pd
from tqdm import tqdm

import json

import numpy as np

def replace_df_values(df, channel_info):
    for key in channel_info:
        value_mappings = channel_info[key]["values"]
        for origin_val in value_mappings:
            mapped_val = value_mappings[origin_val]
            selected_rows = (df[key] == origin_val)

            if np.sum(selected_rows) > 0:
                df.loc[df[key] == origin_val, key] = mapped_val




def load_csv_files_and_get_value_by_attr(file_name, attributes, channel_info):
    df = pd.read_csv(file_name)

    df = df.set_index(['Hours'])

    if attributes is not None:
        selected_df = df[attributes]
    else:
        selected_df = df

    replace_df_values(selected_df, channel_info)

    return selected_df



def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', help='input folder')
    # parser.add_argument('out', help='output folder')
    # parser.add_argument(
    #     '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    # parser.add_argument(
    #     '--show', type=bool, default=False, help='display option')
    # parser.add_argument(
    #     '--wait', type=int, default=0, help='cv2 wait time')
    args = parser.parse_args()
    return args


def retrieve_channel_info():
    with open("channel_info.json") as f:
        channel_info = json.load(f)

    useful_channel_info = {}

    for key in channel_info:
        if len(channel_info[key]["possible_values"]) > 0:
            useful_channel_info[key] = channel_info[key]
    
    return useful_channel_info

def retrieve_value_pairs_for_correlated_attrs(input_folder_name, label_mappings, attributes):
    all_df_ls = []

    all_labels_ls = []

    channel_info = retrieve_channel_info()

    


    # for folder_name in tqdm(os.listdir(input_folder_name)):
    for file_name in tqdm(label_mappings):
        labels = label_mappings[file_name]
        
        # if not file_name.endswith("_timeseries.csv"):
        # full_folder_name = os.path.join(input_folder_name, folder_name)
        # if not os.path.isdir(full_folder_name):
        #     continue
        full_file_name = os.path.join(input_folder_name, file_name)

        if not os.path.exists(full_file_name):
            continue

        # attributes = ["Respiratory rate", "Fraction inspired oxygen"]
        # attributes = ["Systolic blood pressure", "Diastolic blood pressure", "Mean blood pressure"]
        # attributes = ["Pupillary response left", "Pupillary response right", "Glascow coma scale total"]

        selected_df = load_csv_files_and_get_value_by_attr(full_file_name, attributes, channel_info)

        if len(selected_df) > 0:
            all_labels_ls.append(labels)
            all_df_ls.append(selected_df)
            # if all_df_ls is None:
            #     all_df_ls = selected_df
            # else:
            #     all_df_ls = pd.concat([selected_df, all_df_ls], ignore_index=True)

    return all_df_ls, all_labels_ls

def retrieve_value_pairs_for_single_attrs(input_folder_name, label_mappings):
    all_df_ls = []

    all_labels_ls = []

    channel_info = retrieve_channel_info()

    


    # for folder_name in tqdm(os.listdir(input_folder_name)):
    for file_name in tqdm(label_mappings):
        labels = label_mappings[file_name]
        
        # if not file_name.endswith("_timeseries.csv"):
        # full_folder_name = os.path.join(input_folder_name, folder_name)
        # if not os.path.isdir(full_folder_name):
        #     continue
        full_file_name = os.path.join(input_folder_name, file_name)

        if not os.path.exists(full_file_name):
            continue

        # attributes = ["Respiratory rate", "Fraction inspired oxygen"]
        # attributes = ["Systolic blood pressure", "Diastolic blood pressure", "Mean blood pressure"]
        # attributes = ["Pupillary response left", "Pupillary response right", "Glascow coma scale total"]

        selected_df = load_csv_files_and_get_value_by_attr(full_file_name, None, channel_info)

        if len(selected_df) > 0:
            all_labels_ls.append(labels)
            all_df_ls.append(selected_df)
            # if all_df_ls is None:
            #     all_df_ls = selected_df
            # else:
            #     all_df_ls = pd.concat([selected_df, all_df_ls], ignore_index=True)

    return all_df_ls, all_labels_ls





def main():
    args = parse_args()

    attributes = ["Systolic blood pressure", "Diastolic blood pressure", "Mean blood pressure"]

    retrieve_value_pairs_for_correlated_attrs(args, attributes)    

    print() 

    
    # file_name = os.path.join(args.input, "10013_episode1_timeseries.csv")

    


if __name__ == '__main__':
    main()
