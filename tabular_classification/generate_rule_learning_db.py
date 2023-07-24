import os, sys
import numpy as np
import argparse
import pandas as pd
import json
from tabular_classification.pre_processing import read_tab_data, continue_feats, discrete_feats, bin_discrete_feats, target_col

def output_facts_to_file(file_name, sample_id_ls):
    with open(file_name, "w") as f:
        for sample_id in sample_id_ls:
            f.write(str(sample_id) + "\n")

        f.close()


def output_facts_for_continuous_attribute(ds, attr, out_dir):
    percentile_ls = [5, 20, 35, 50, 65, 80, 95]

    array = np.array(ds[attr])
    p_ls = []
    for percentile in percentile_ls:
        
        p = np.percentile(array, percentile)
        p_ls.append(p)

    print("splits values for attribute ", attr)
    print(p_ls)
    
    schema_ls = []
    total_sample_count = len(ds)
    real_sample_count = 0

    for p_idx in range(len(p_ls) + 1):
        relation_name = attr + "_" + str(p_idx)
        schema_ls.append(relation_name)
        output_f_name = os.path.join(out_dir, relation_name + ".facts")
        if p_idx == 0:
            p = p_ls[p_idx]
            sample_id_ls = np.nonzero(array <= p)[0]
        elif p_idx == len(p_ls):
            p = p_ls[p_idx-1]
            sample_id_ls = np.nonzero(array > p)[0]
        else:
            p1 = p_ls[p_idx-1]
            p2 = p_ls[p_idx]
            
            sample_id_ls = np.nonzero(((array > p1)&(array <= p2)))[0]
        real_sample_count += len(sample_id_ls)
        output_facts_to_file(output_f_name, sample_id_ls.reshape(-1))

    assert total_sample_count == real_sample_count
    return schema_ls, p_ls

def output_facts_for_discrete_attribute(ds, attr, out_dir):
    ds_array = np.array(ds[attr])
    unique_val_ls = np.unique(ds_array)

    schema_ls = []
    for val in unique_val_ls:
        relation_name = attr + "_" + str(val)
        schema_ls.append(relation_name)
        output_f_name = os.path.join(out_dir, relation_name + ".facts")
        sample_id_ls = np.nonzero(ds_array == val)[0]
        output_facts_to_file(output_f_name, sample_id_ls.reshape(-1))

    return schema_ls, unique_val_ls.tolist()


def output_facts_for_labels(ds, attr, out_dir):
    ds_array = np.array(ds[attr])
    # unique_val_ls = ds_array.unique()

    schema_ls = []
    # for val in unique_val_ls:
    for val in [0,1]:
        if val == 0:
            relation_name = "not_has"
        else:
            relation_name = "has"
        schema_ls.append(relation_name)
        output_f_name = os.path.join(out_dir, relation_name + ".facts")
        sample_id_ls = np.nonzero(ds_array == val)[0]
        output_facts_to_file(output_f_name, sample_id_ls.reshape(-1))

    return schema_ls

def output_schema_ls(schema_ls, out_dir):
    output_file_name = os.path.join(out_dir, "master.dl")

    with open(output_file_name, "w") as f:
        for idx in range(len(schema_ls)):
            if idx < len(schema_ls) - 2:
                relation = schema_ls[idx]
                f.write(".input " + relation + "(v:ID)\n")
            else:
                relation = schema_ls[idx]
                f.write(".output " + relation + "(v:ID)\n")

        f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    # parser.add_argument('--epochs', type=int, default=200, help='used for resume')
    # # parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    # # parser.add_argument('--lr', type=float, default=0.02, help='used for resume')

    # parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    # parser.add_argument('--lr', type=float, default=0.002, help='used for resume')
    # parser.add_argument('--model', type=str, default=0.02, help='used for resume', choices=["mlp", "dd"])
    # parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--work_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--data_dir', type=str, default="out/", help='used for resume')

    args = parser.parse_args()
    return args


def main(args):
    file_name = os.path.join(args.data_dir, "cardio_train.csv")
    ds = read_tab_data(file_name)
    db_instance_dir = os.path.join(args.work_dir, "db_instance/")
    db_schema_dir = os.path.join(args.work_dir, "db_schema/")

    os.makedirs(db_instance_dir, exist_ok=True)
    os.makedirs(db_schema_dir, exist_ok=True)


    all_schema_ls = []
    split_mappings = dict()
    for attr in continue_feats:
        schema_ls, p_ls = output_facts_for_continuous_attribute(ds, attr, db_instance_dir)        
        split_mappings[attr] = p_ls
        all_schema_ls.extend(schema_ls)
    
    for attr in discrete_feats:
        schema_ls, unique_val_ls = output_facts_for_discrete_attribute(ds, attr, db_instance_dir)        
        split_mappings[attr] = unique_val_ls
        all_schema_ls.extend(schema_ls)

    for attr in bin_discrete_feats:
        schema_ls, unique_val_ls = output_facts_for_discrete_attribute(ds, attr, db_instance_dir)        
        split_mappings[attr] = unique_val_ls
        all_schema_ls.extend(schema_ls)

    for attr in target_col:
        schema_ls = output_facts_for_labels(ds, attr, db_instance_dir)        
        all_schema_ls.extend(schema_ls)

    with open(os.path.join(args.data_dir, "dataframe_split.json"), "w") as f:
        json.dump(split_mappings, f)
        f.close()        

    output_schema_ls(all_schema_ls, db_schema_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

