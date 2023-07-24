import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from mTan_dataset import variable_time_collate_fn
import torch
from tqdm import tqdm
# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1, root_data_path=None):
    if root_data_path is None:
        # data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
        data = pd.read_csv("./data/set-a/{}.txt".format(id_))
    else:
        # data = pd.read_csv(os.path.join(root_data_path, "data/physio/set-a/{}.txt").format(id_))
        data = pd.read_csv(os.path.join(root_data_path, "data/set-a/{}.txt").format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def get_idlist(root_data_path):
    patient_id = []
    if root_data_path is None:
        data_path = "./data/set-a"
    else:
        data_path = os.path.join(root_data_path, "data/set-a")
        
    for filename in os.listdir(data_path):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0, root_data_path=None, data_max=None, data_min=None):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.origin_observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        if root_data_path is None:
            path = (
                "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
            )
        else:
            path = (
                os.path.join(root_data_path, "data/physio_missing") + str(missing_ratio) + "_seed" + str(seed) + ".pk"
            )

        # if os.path.isfile(path) == False:  # if datasetfile is none, create
        idlist = get_idlist(root_data_path)
        for id_ in tqdm(idlist):
            try:
                observed_values, observed_masks, gt_masks = parse_id(
                    id_, missing_ratio, root_data_path=root_data_path
                )
                self.origin_observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
            except Exception as e:
                print(id_, e)
                continue
        self.origin_observed_values = torch.from_numpy(np.array(self.origin_observed_values))
        self.observed_masks = torch.from_numpy(np.array(self.observed_masks))
        self.gt_masks = torch.from_numpy(np.array(self.gt_masks))

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = self.origin_observed_values.reshape(-1, 35)
        tmp_masks = self.observed_masks.reshape(-1, 35)
        # mean = np.zeros(35)
        # std = np.zeros(35)
        if data_max is None:
            data_max = np.zeros(35)
        if data_min is None:
            data_min = np.zeros(35)
        for k in range(35):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            data_max[k] = c_data.max()
            data_min[k] = c_data.min()
        # self.observed_values = (
        #     (self.observed_values - mean) / std * self.observed_masks
        # )
        self.observed_values = (self.origin_observed_values - data_min)/(data_max - data_min)
        self.data_max = data_max
        self.data_min = data_min

        with open(path, "wb") as f:
            pickle.dump(
                [self.origin_observed_values, self.observed_values, self.observed_masks, self.gt_masks, self.data_max, self.data_min], f
            )
        # else:  # load datasetfile
        #     with open(path, "rb") as f:
        #         self.origin_observed_values, self.observed_values, self.observed_masks, self.gt_masks, self.data_max, self.data_min = pickle.load(
        #             f
        #         )
        #         if not type(self.observed_values) is torch.Tensor:
        #             self.observed_values = torch.from_numpy(self.observed_values)
        #         if not type(self.observed_masks) is torch.Tensor:
        #             self.observed_masks = torch.from_numpy(self.observed_masks)
        #         if not type(self.gt_masks) is torch.Tensor:
        #             self.gt_masks = torch.from_numpy(self.gt_masks)
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        
        # if self.all_test_mask_ls is not None:
        return self.observed_values[index], self.observed_masks[index], torch.arange(self.eval_length), self.gt_masks[index]
        # else:
        #     return self.observed_values[index], self.observed_masks[index], self.time_step_ls[index], self.label_tensor_ls[index], self.data_min, self.data_max, None

        # s = {
        #     "observed_data": self.observed_values[index],
        #     "observed_mask": self.observed_masks[index],
        #     "gt_mask": self.gt_masks[index],
        #     "timepoints": np.arange(self.eval_length),
        # }
        # return s

    @staticmethod
    def collate_fn(data):
        time_step_ls = [item[2] for item in data]
        visit_tensor_ls = [item[0] for item in data]
        mask_ls = [item[1] for item in data]
        # label_tensor_ls = [item[3] for item in data]
        # person_info_ls = [item[3].view(-1) for item in data]
        
        test_mask_ls = None
        if data[0][3] is not None:
            test_mask_ls = [item[3] for item in data]
            
        batched_data_tensor, combined_test_mask = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, device=torch.device("cpu"), data_min=None, data_max=None, test_mask_ls = test_mask_ls)
        # batched_person_=[]
        # batched_person_info = torch.stack(person_info_ls)
        # batched_data_tensor = torch.stack([item[0] for item in data])
        # batched_label_tensor = torch.stack([item[1] for item in data])
        return batched_data_tensor, combined_test_mask

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(root_data_path=None, seed=1, nfold=0, batch_size=16, missing_ratio=0.2, do_train=False):
    if do_train:
        # only to obtain total length of dataset
        dataset = Physio_Dataset(missing_ratio=0.0, seed=seed, root_data_path=root_data_path)
        torch.save(dataset, os.path.join(root_data_path, "dataset"))
        indlist = np.arange(len(dataset))

        np.random.seed(seed)
        np.random.shuffle(indlist)

        # 5-fold test
        start = (int)(nfold * 0.2 * len(dataset))
        end = (int)((nfold + 1) * 0.2 * len(dataset))
        test_index = indlist[start:end]
        remain_index = np.delete(indlist, np.arange(start, end))

        np.random.seed(seed)
        np.random.shuffle(remain_index)
        num_train = (int)(len(dataset) * 0.7)
        train_index = remain_index[:num_train]
        valid_index = remain_index[num_train:]

        dataset = Physio_Dataset(
            use_index_list=train_index, missing_ratio=0.0, seed=seed, root_data_path=root_data_path
        )
        torch.save(dataset, os.path.join(root_data_path, "train_dataset"))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1, collate_fn=Physio_Dataset.collate_fn)
        valid_dataset = Physio_Dataset(
            use_index_list=valid_index, missing_ratio=0.1, seed=seed, data_max=dataset.data_max, data_min=dataset.data_min, root_data_path=root_data_path
        )
        torch.save(valid_dataset, os.path.join(root_data_path, "valid_dataset"))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0, collate_fn=Physio_Dataset.collate_fn)
        test_dataset = Physio_Dataset(
            use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, data_max=dataset.data_max, data_min=dataset.data_min, root_data_path=root_data_path
        )
        torch.save(test_dataset, os.path.join(root_data_path, "test_dataset"))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0, collate_fn=Physio_Dataset.collate_fn)
        
    else:
        dataset = torch.load(os.path.join(root_data_path, "train_dataset"))
        valid_dataset = torch.load(os.path.join(root_data_path, "valid_dataset"))
        test_dataset = torch.load(os.path.join(root_data_path, "test_dataset"))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1, collate_fn=Physio_Dataset.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0, collate_fn=Physio_Dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0, collate_fn=Physio_Dataset.collate_fn)
    return dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, dataset.data_max, dataset.data_min