import pickle
import pandas as pd
def attr_ls_to_str(attr_ls):
    attr_str = ""
    for k in range(len(attr_ls)):
        if k >= 1:
            attr_str += ","

        attr_str += attr_ls[k]
    return attr_str

def save_objs(obj, file_name):
    with open(file_name, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)

def save_df_ls(obj_ls, file_name):
    f_name_ls_str = ""
    for idx in range(len(obj_ls)):
        if idx >= 1:
            f_name_ls_str += ","
        obj = obj_ls[idx]
        obj.to_csv(file_name + "_" + str(idx) + ".csv")
        f_name_ls_str += file_name + "_" + str(idx) + ".csv"
    with open(file_name, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(f_name_ls_str, outp)


def load_objs(file_name):
    with open(file_name, 'rb') as f:
        load_obj = pickle.load(f)
    return load_obj

def load_df_ls(file_name):
    with open(file_name, 'rb') as outp:  # Overwrites any existing file.
        f_name_ls_str = pickle.load(outp)
        
    f_name_ls = f_name_ls_str.split(",")
    obj_ls = []
    for f_name in f_name_ls:
        df = pd.read_csv(f_name)
        obj_ls.append(df)
    return obj_ls


def read_correlated_attributes(file_name):
    coorelated_attr_pair_ls = []
    with open(file_name) as f:
        for line in f:
            attr1, attr2 = line.split(",")
            attr1 = attr1.strip()
            attr2 = attr2.strip()
            coorelated_attr_pair_ls.append(tuple((attr1, attr2)))

    return coorelated_attr_pair_ls


def test_numeric(array):
    dtype = str(array.dtype)
    
    return "int" in dtype or "float" in dtype or "long" in dtype or "double" in dtype