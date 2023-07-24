import pandas as pd


def read_in_hospital_mortality_label_file(file_name):
    label_df = pd.read_csv(file_name)
    label_mappings = {}

    for idx in range(len(label_df)):
        file_name = label_df.loc[idx, "stay"]
        label = label_df.loc[idx, "y_true"]
        label_mappings[file_name] = label
    return label_mappings