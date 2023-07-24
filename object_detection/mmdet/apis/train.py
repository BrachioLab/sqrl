import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner

from mmdet.core import (DistEvalHook, DistOptimizerHook, EvalHook,
                        Fp16OptimizerHook, build_optimizer, build_optimizer_tent)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from .test import output_gt_bbox_for_comparison

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from baseline_methods import l2_consistency_loss
import Norm.Norm as norm

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    # print(f"Parsing losses in {losses}")
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            if len(loss_value) > 0:
                log_vars[loss_name] = torch.clamp(sum(_loss.mean() for _loss in loss_value), max=500.00)
            else:
                log_vars[loss_name] = torch.tensor(0.0, requires_grad=True)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    # print(f"Adding the following losses", [(_key, _value) for _key, _value in log_vars.items() if 'loss' in _key])

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if torch.is_tensor(loss_value):
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))

            log_vars[loss_name] = loss_value.item()
        else:
            log_vars[loss_name] = loss_value

    if torch.isnan(loss):
        print("MASSIVE BLOODY ISSUE")
        exit(-1)

    return loss, log_vars


    # data['test_adaptations'] = tta_method
def batch_processor(model, data, train_mode, test_adaptations=False, tta_method='rule'):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    if not type(data) is list:
        data['test_adaptations'] = test_adaptations
        data['tta_method'] = tta_method
        losses = model(**data)
        loss, log_vars = parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
    else:
        all_output = []
        for item in data:
            item['test_adaptations'] = test_adaptations
            item['tta_method'] = tta_method
            output = model(**item)
            all_output.append(output)

        loss = l2_consistency_loss(all_output)
        loss, log_vars = parse_losses({"memo_loss":loss})
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img'].data))
    

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None, test_adaptations = False):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta, test_adaptations = test_adaptations)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta, test_adaptations = test_adaptations)


def _dist_train(model,
                dataset,
                cfg,
                validate=False,
                logger=None,
                timestamp=None,
                meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(DistEvalHook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None, test_adaptations = False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    print(dataset)
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    
    output_gt_bbox_for_comparison(cfg.work_dir, data_loaders[0])
    # build runner
    if not test_adaptations:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        optimizer = build_optimizer(model, cfg.optimizer)
    else:
        
        optimizer,model = build_optimizer_tent(model, cfg.optimizer, cfg.full_model)
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register eval hooks
    if validate:
        print("VALIDATING")
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # data_loaders.append(val_dataloader)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    print(len(data_loaders))
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, test_adaptations = test_adaptations, tta_method = cfg.tta_method)
