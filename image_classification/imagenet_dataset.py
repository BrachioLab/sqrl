from torchvision import models
import torch
from torchvision import transforms
import sys
from PIL import Image
from imagenet_classes import classes
import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Tuple
import json
from imagenet_x import load_annotations
import pandas as pd
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from baseline_methods import image_aug

class MyImageNet(ImageFolder):
    def __init__(self, image_dir, split='val', use_annotation = False, transform=None, class_id_meta_class_mappings = None, meta_class_id_mappings = None, new_path=None):
        super().__init__(image_dir, transform=transform)
        self.meta_class_mappings = class_id_meta_class_mappings
        self.meta_class_id_mappings = meta_class_id_mappings
        if use_annotation:
            self.annotations = load_annotations(partition=split)
        else:
            self.annotations = None

        self.new_path=new_path


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        file_name = path.split("/")[-1]

        if self.annotations is not None:
            annotation_df = self.annotations[self.annotations["file_name"] == file_name]
        else:
            annotation_df = None

        # target = list(annotation_df["metaclass"])[0]
        target = self.meta_class_id_mappings[self.meta_class_mappings[target]]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, annotation_df

    def init_samples(self, indices, split='val'):
        sample_ls = []
        target_ls = []
        df_ls = []
        path_ls = []
        if self.annotations is None:
            annotations = load_annotations(partition=split)
        else:
            annotations = self.annotations

        for index in tqdm(indices):
            path, target = self.samples[index]
            target = self.meta_class_id_mappings[self.meta_class_mappings[target]]
            sample = None # self.loader(path)
            file_name = path.split("/")[-1]

            # if self.annotations is not None:
            annotation_df = annotations[annotations["file_name"] == file_name]
            # else:
            #     annotation_df = None

            sample_ls.append(sample)
            target_ls.append(target)
            df_ls.append(annotation_df)
            path_ls.append(path)

        return sample_ls, target_ls, df_ls, path_ls

    @staticmethod
    def collate_fn(data):
        if type(data[0][0]) is not list:
            sample_ls = [data[idx][0] for idx in range(len(data))]
            sample_tensor = torch.stack(sample_ls, dim = 0)
        else:
            sample_tensor = []
            for aug_idx in range(len(data[0][0])):
                sample_ls = [data[idx][0][aug_idx] for idx in range(len(data))]
                sub_sample_tensor = torch.stack(sample_ls, dim = 0)
                sample_tensor.append(sub_sample_tensor)
        target_ls = [data[idx][1] for idx in range(len(data))]
        path_ls = [data[idx][2] for idx in range(len(data))]
        if data[0][3] is not None:
            annotation_ls = [data[idx][3] for idx in range(len(data))]
            annotation_tensor = pd.concat(annotation_ls)
        else:
            annotation_tensor = None
        
        target_tensor = torch.tensor(target_ls)
        
        return sample_tensor, target_tensor, path_ls, annotation_tensor



class MyImageNet_test(Dataset):
    def __init__(self, sample_ls, target_ls, df_ls, path_ls, transform=None, target_transform=None, augment=False, preprocess=None, loader = default_loader, new_path=None):
        # super().__init__(image_dir, transform=transform)
        # self.meta_class_mappings = class_id_meta_class_mappings
        # self.meta_class_id_mappings = meta_class_id_mappings

        self.sample_ls = sample_ls
        self.target_ls = target_ls
        self.df_ls = df_ls
        self.path_ls = path_ls
        self.transform = transform
        self.target_transform = target_transform
        self.augment=augment
        self.preprocess = preprocess
        self.turn_off_aug = False
        self.loader = loader
        self.new_path = new_path

    

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path, target = self.samples[index]

        
        # # target = list(annotation_df["metaclass"])[0]
        # target = self.meta_class_id_mappings[self.meta_class_mappings[target]]

        # sample = self.loader(path)
        # sample = self.sample_ls[index]

        target = self.target_ls[index]
        annotation_df = self.df_ls[index]
        path = self.path_ls[index]
        if self.new_path is not None:
            file_name = path.split("/")[-1]
            sub_folder_name = path.split("/")[-2]
            path = os.path.join(self.new_path, sub_folder_name, file_name)

        if hasattr(self, 'loader'):
            sample = self.loader(path)
        else:
            sample = self.sample_ls[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if hasattr(self, 'augment') and self.augment:
            if not self.turn_off_aug:
                sample = [self.preprocess(sample), image_aug(sample, self.preprocess), image_aug(sample, self.preprocess)]
            else:
                sample = self.preprocess(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, annotation_df
    def __len__(self):
        return len(self.sample_ls)

    @staticmethod
    def collate_fn(data):
        sample_ls = [data[idx][0] for idx in range(len(data))]
        target_ls = [data[idx][1] for idx in range(len(data))]
        path_ls = [data[idx][2] for idx in range(len(data))]
        if data[0][3] is not None:
            annotation_ls = [data[idx][3] for idx in range(len(data))]
            annotation_tensor = pd.concat(annotation_ls)
        else:
            annotation_tensor = None
        sample_tensor = torch.stack(sample_ls, dim = 0)
        target_tensor = torch.tensor(target_ls)
        
        return sample_tensor, target_tensor, path_ls, annotation_tensor