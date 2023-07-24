""" tools.py
    Utility functions that are common to all tasks
    Developed for Tabular-Transfer-Learning project
    March 2022
"""

import logging
import os
import random
from collections import OrderedDict
import torch
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR, ChainedScheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch import Tensor

import deep_tabular.models as models
from .data_tools import get_data, get_categories_full_cat_data, TabularDataset, get_multilabel_data
from .warmup import ExponentialWarmup, LinearWarmup
from ..adjectives import adjectives
from ..names import names
from .mimic_tools import data_prep_transfer_mimic
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))


import Tent.Tent as tent
import Norm.Norm as norm

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115

def generate_run_id():
    hashstr = f"{adjectives[random.randint(0, len(adjectives))]}-{names[random.randint(0, len(names))]}"
    return hashstr


def write_to_tb(stats, stat_names, epoch, writer):
    for name, stat in zip(stat_names, stats):
        stat_name = os.path.join("val", name)
        writer.add_scalar(stat_name, stat, epoch)


class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors: Tensor, df =None, augment=False) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.df = df
        self.augment=augment

    def __getitem__(self, index):
        return_data = []
        for tensor in self.tensors:
            return_data.append(tensor[index])
        
        # return tuple(tensor[index] for tensor in self.tensors)
        return return_data, self.df.iloc[index]

    def __len__(self):
        return self.tensors[0].size(0)
    
    @staticmethod
    def collate_func(data):

        
        df_ls = [item[1] for item in data]

        df_full = pd.concat(df_ls)


        return_data = []
        for tensor_idx in range(len(data[0][0])):
            return_data.append(torch.stack([item[0][tensor_idx] for item in data]))

        return_data.append(df_full)
        return tuple(return_data)

def get_dataloaders(cfg, which_dataset=None):
    """
    cfg: OmegaConf, dictionary of configurations
    which_dataset: int of None, indicates which dataset to use if multiple are specified in the config
    """

    if which_dataset is not None:
        cfg_dataset = cfg.dataset[which_dataset]
    else:
        cfg_dataset = cfg.dataset

    if cfg_dataset.task == 'multilabel':
        #Changing the task to binclass because multilabel is just binary cross-entropy over multilabel logits
        cfg_dataset.task = 'binclass'
        x_numerical, x_categorical, y, info, full_cat_data_for_encoder, full_df = get_multilabel_data(ds_id=cfg_dataset.name,
                                                                                             source=cfg_dataset.source,
                                                                                             task=cfg_dataset.task, path=cfg_dataset.data_path)
    elif cfg_dataset.name == 'mimic':
        x_numerical, x_categorical, y, info, full_cat_data_for_encoder, full_df = data_prep_transfer_mimic(ds_id=cfg_dataset.name,
                                                                                                 task=cfg_dataset.task,
                                                                                                 stage=cfg_dataset.stage,
                                                                                                 downstream_target=cfg_dataset.downstream_target,
                                                                                                 downstream_samples_per_class=cfg_dataset.downstream_sample_num//2)
    else:
        x_numerical, x_categorical, y, info, full_cat_data_for_encoder, full_df = get_data(path=cfg_dataset.data_path,dataset_id=cfg_dataset.name,
                                                                                  source=cfg_dataset.source,
                                                                                  task=cfg_dataset.task,
                                                                                  datasplit=[.65, .15, .2])

    dataset = TabularDataset(x_numerical, x_categorical, y, info, normalization=cfg_dataset.normalization,
                             cat_policy="indices",
                             seed=0,
                             full_cat_data_for_encoder=full_cat_data_for_encoder,
                             full_df = full_df,
                             y_policy=cfg_dataset.y_policy,
                             normalizer_path=cfg_dataset.normalizer_path,
                             stage=cfg_dataset.stage)

    X = dataset.preprocess_data()
    Y, y_info = dataset.build_y()
    unique_categories = get_categories_full_cat_data(full_cat_data_for_encoder)
    n_numerical = dataset.n_num_features
    n_categorical = dataset.n_cat_features
    n_classes = dataset.n_classes
    logging.info(f"Task: {cfg_dataset.task}, Dataset: {cfg_dataset.name}, n_numerical: {n_numerical}, "
                 f"n_categorical: {n_categorical}, n_classes: {n_classes}, n_train_samples: {dataset.size('train')}, "
                 f"n_val_samples: {dataset.size('val')}, n_test_samples: {dataset.size('test')}")

    trainset = TensorDataset(X[0]["train"], X[1]["train"], Y["train"], df=full_df["train_df"])
    valset = TensorDataset(X[0]["val"], X[1]["val"], Y["val"], df=full_df["val_df"])
    testset = TensorDataset(X[0]["test"], X[1]["test"], Y["test"], df=full_df["test_df"])
    if not cfg.dataset.train and cfg.hyp.tta_method == "memo":
        testset = TensorDataset(X[0]["test"], X[1]["test"], Y["test"], df=full_df["test_df"], augment=True)


    trainloader = DataLoader(trainset, batch_size=cfg.hyp.train_batch_size, shuffle=True, drop_last=True, collate_fn=TensorDataset.collate_func)
    valloader = DataLoader(valset, batch_size=cfg.hyp.test_batch_size, shuffle=False, drop_last=False, collate_fn=TensorDataset.collate_func)
    testloader = DataLoader(testset, batch_size=cfg.hyp.test_batch_size, shuffle=False, drop_last=False, collate_fn=TensorDataset.collate_func)


    sampler = RandomSampler(testset, replacement=True, num_samples=len(testset)*4)
    test_adapt_eval_loader = DataLoader(testset, sampler=sampler, batch_size=cfg.hyp.test_batch_size, collate_fn=TensorDataset.collate_func)

    test_adapt_loader = DataLoader(testset, batch_size=cfg.hyp.test_batch_size, shuffle=True, drop_last=True, collate_fn=TensorDataset.collate_func)

    loaders = {"train": trainloader, "val": valloader, "test": testloader, "test_adapt": test_adapt_loader, "test_adapt_eval": test_adapt_eval_loader}

    if cfg.hyp.validate:
        sampler = RandomSampler(trainset, replacement=True, num_samples=len(trainset)*2)
        train_rule_eval_loader = DataLoader(trainset, collate_fn = TensorDataset.collate_func,sampler=sampler, batch_size=cfg.hyp.test_batch_size)

        sampler = RandomSampler(valset, replacement=True, num_samples=len(valset)*2)
        valid_rule_eval_loader = DataLoader(valset, collate_fn = TensorDataset.collate_func,sampler=sampler, batch_size=cfg.hyp.test_batch_size)
        loaders["train_rule_eval"] = train_rule_eval_loader
        loaders["valid_rule_eval"] = valid_rule_eval_loader


    return loaders, unique_categories, n_numerical, n_classes


def get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_params):
    model = model.lower()
    net = getattr(models, model)(num_numerical, unique_categories, num_outputs, d_embedding, model_params)
    return net


def get_embedder(cfg, num_numerical, unique_categories):
    model_name = cfg.model.name.lower()
    if model_name == "ft_transformer":
        embedder = models.ft_tokenizer(num_numerical, unique_categories, cfg.model.d_embedding, cfg.model.token_bias)
    else:
        raise NotImplementedError(f"Model name is {model_name}, but this is not yet implemented.")
    return embedder


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)



def get_backbone(cfg, device):
    model_name = cfg.model.name.lower()
    if model_name == "ft_transformer":
        net = models.ft_backbone(cfg.model)
    else:
        raise NotImplementedError(f"Model name is {model_name}, but this is not yet implemented.")
    if cfg.model.model_path is not None:
        logging.info(f"Loading backbone from checkpoint {cfg.model.model_path}...")
        state_dict = torch.load(cfg.model.model_path, map_location=device)
        net.load_state_dict(state_dict["backbone"])
    net = net.to(device)
    return net


def get_optimizer_for_single_net(optim_args, net, state_dict, full_model=False):
    warmup = ExponentialWarmup if optim_args.warmup_type == "exponential" else LinearWarmup

    if optim_args.tta_method == "norm":
        net = norm.Norm(net)
        params = net.parameters()
        all_params = [{"params": params}]
    else:
        if full_model:
            if optim_args.head_lr is not None:
                head_name, head_module = list(net.named_modules())[-1]
                head_parameters = [v for k, v in net.named_parameters() if head_name in k]
                feature_extractor_parameters = [v for k, v in net.named_parameters() if head_name not in k]
                all_params = [{'params': feature_extractor_parameters},
                            {'params': head_parameters, 'lr': optim_args.head_lr}]
            else:
                all_params = [{"params": [p for n, p in net.named_parameters()]}]
        else:
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            all_params = [{"params": params}]


    if optim_args.optimizer.lower() == "sgd":
        optimizer = SGD(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay,
                        momentum=optim_args.momentum)
    elif optim_args.optimizer.lower() == "adam":
        optimizer = Adam(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    elif optim_args.optimizer.lower() == "adamw":
        optimizer = AdamW(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choice of {optim_args.optimizer.lower()} not yet implmented. "
                         f"Should be one of ['sgd', 'adam', 'adamw'].")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = warmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup(optimizer, warmup_period=optim_args.warmup_period)

    if optim_args.lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=optim_args.lr_schedule,
                                   gamma=optim_args.lr_factor, last_epoch=-1)
    elif optim_args.lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, optim_args.epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {optim_args.lr_decay} not yet implemented.")

    #Freeze feature extractor and warm the head for some period
    if optim_args.head_warmup_period is not None and full_model:
        #Multiply the feature extractor lr by 0 during the head warmup period
        lambda_feature_extractor = lambda epoch: 0 if epoch < optim_args.head_warmup_period else 1
        lambda_head = lambda epoch: 1
        head_warmup_scheduler = LambdaLR(optimizer, lr_lambda = [lambda_feature_extractor, lambda_head])
        lr_scheduler = ChainedScheduler([head_warmup_scheduler, lr_scheduler])
    return optimizer, warmup_scheduler, lr_scheduler

# def build_optimizer_tent(model, optimizer_cfg, full_model):
    
#     if not full_model:
#         model = tent.configure_model(model)
#         optimizer_cfg = optimizer_cfg.copy()
#         paramwise_options = optimizer_cfg.pop('paramwise_options', None)
#         params, param_names = tent.collect_params(model)
#         optimizer_cfg['params'] = params
#         optimizer = build_from_cfg(optimizer_cfg, OPTIMIZERS)
#         tented_model = tent.Tent(model, optimizer)
#         return optimizer, tented_model.model
#     else:
#         model = tent.configure_model2(model)
#         optimizer_cfg['params'] = model.parameters()
#         optimizer = build_from_cfg(optimizer_cfg, OPTIMIZERS)
#         # tented_model = tent.Tent(model, optimizer)
#         return optimizer, model

def get_optimizer_for_backbone(optim_args, embedders, backbone, heads, state_dict=None):
    warmup = ExponentialWarmup if optim_args.warmup_type == "exponential" else LinearWarmup

    all_params = [{"params": [p for p in backbone.parameters()], "lr": optim_args.lr}]
    all_params.extend([{f"params": [p for p in v.parameters()],
                        "lr": optim_args.lr_for_embedders} for v in embedders.values()])
    all_params.extend([{f"params": [p for p in v.parameters()],
                        "lr": optim_args.lr_for_heads} for v in heads.values()])

    if optim_args.optimizer.lower() == "adamw":
        optimizer = AdamW(all_params, weight_decay=optim_args.weight_decay)
    elif optim_args.optimizer.lower() == "sgd":
        optimizer = SGD(all_params, momentum=0.9, weight_decay=optim_args.weight_decay)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choice of {optim_args.optimizer.lower()} not yet implmented. "
                         f"Should be one of ['adamw'].")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = warmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup(optimizer, warmup_period=optim_args.warmup_period)

    if optim_args.lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=optim_args.lr_schedule,
                                   gamma=optim_args.lr_factor, last_epoch=-1)
    elif optim_args.lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, optim_args.epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {optim_args.lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler


def get_criterion(task):
    if task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    elif task == "binclass":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "regression":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"No loss function implemented for task {task}.")
    return criterion

def get_head(model_name, net):
    if model_name in ['ft_transformer', 'resnet', 'mlp']:
        head_name = 'head'
        head_module = net.head
    else:
        head_name, head_module = list(net.named_modules())[-1]
    print(f'Original head: {head_name}, {head_module}\n')
    return head_name, head_module

def remove_parallel(state_dict):
    ''' state_dict: state_dict of model saved with DataParallel()
        returns state_dict without extra module level '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict

def load_transfer_model_from_checkpoint(model_args, num_numerical, unique_categories, num_outputs, device):
    model = model_args.name
    model_path = model_args.model_path
    d_embedding = model_args.d_embedding
    use_mlp_head = model_args.use_mlp_head
    freeze_feature_extractor = model_args.freeze_feature_extractor
    epoch = 0
    optimizer = None

    net = get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_args)
    net = net.to(device)
    head_name, head_module = get_head(model_args.name, net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        if device == "cuda":
            state_dict["net"] = remove_parallel(state_dict["net"])
        pretrained_feature_extractor_dict = {k: v for k, v in state_dict["net"].items() if head_name not in k}
        missing_keys, unexpected_keys = net.load_state_dict(pretrained_feature_extractor_dict, strict = False)
        print('State dict successfully loaded from pretrained checkpoint. Original head reinitialized.')
        print('Missing keys:{}\nUnexpected keys:{}\n'.format(missing_keys, unexpected_keys))
        # epoch = state_dict["epoch"] + 1
        # optimizer = state_dict["optimizer"]
    if freeze_feature_extractor:
        trainable_params = []
        for name, param in net.named_parameters():
            if not any(x in name for x in [head_name]):
                # if head_name not in name:
                param.requires_grad = False
            else:
                trainable_params.append(name)
        print(f'Feature extractor frozen. Trainable params: {trainable_params}')

    if use_mlp_head:
        emb_dim = head_module.in_features
        out_dim = head_module.out_features
        head_module = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, out_dim)).to(device)
        setattr(net, head_name, head_module)
    print('New head set to:', net.head)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    return net, epoch, optimizer


def load_model_from_checkpoint(model_args, num_numerical, unique_categories, num_outputs, device):
    model = model_args.name
    model_path = model_args.model_path
    d_embedding = model_args.d_embedding
    epoch = 0
    optimizer = None

    net = get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_args)
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]

    return net, epoch, optimizer

