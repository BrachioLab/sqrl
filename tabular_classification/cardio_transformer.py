""" train_net_from_scratch.py
    Train, test, and save neural networks without transfer learning
    Developed for Tabular Transfer Learning project
    March 2022
"""
import random
import json
import logging
import os, sys
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
# from torch.utils.tensorboard import SummaryWriter

import deep_tabular as dt
import argparse
import pickle
from tqdm import tqdm
from deep_tabular.utils.test_time_adaptation import learning_statistics_for_bp_rule
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Tent.utils import *
from rule_processing.sigmoidF1 import sigmoidF1
from rule_processing.process_rules import *
from baseline_methods import available_tta_methods
# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115

input_rule_name = "rule_f1_bounds_cardiovascular.jsonl"

@hydra.main(config_path="config", config_name="train_net_config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # args = parse_args()    
    torch.backends.cudnn.benchmark = True
    cfg.dataset.train = bool(cfg.dataset.train)
    cfg.dataset.full_model = bool(cfg.dataset.full_model)
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_net_from_scratch.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.hyp.save_period < 0:
        cfg.hyp.save_period = 1e8
    torch.manual_seed(cfg.hyp.seed)
    torch.cuda.manual_seed_all(cfg.hyp.seed)
    random.seed(cfg.hyp.seed)
    np.random.seed(cfg.hyp.seed)
    # writer = SummaryWriter(log_dir=f"tensorboard")
    if not cfg.dataset.train:
        if cfg.dataset.full_model:
            output_path = os.path.join(cfg.dataset.out_path, cfg.model.name + "_" + cfg.hyp.tta_method + "_full_model")
        else:
            output_path = os.path.join(cfg.dataset.out_path, cfg.model.name + "_" + cfg.hyp.tta_method)
        cache_path = os.path.join(cfg.dataset.out_path, cfg.model.name)
    else:
        output_path = os.path.join(cfg.dataset.out_path, cfg.model.name)
        cache_path = output_path
        
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)
    ####################################################
    #               Dataset and Network and Optimizer
    loaders, unique_categories, n_numerical, n_classes = dt.utils.get_dataloaders(cfg)

    meta_class_mappings = {"has":1, "not_has":0}

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.model,
                                                                                 n_numerical,
                                                                                 unique_categories,
                                                                                 n_classes,
                                                                                 device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    log.info(f"This {cfg.model.name} has {pytorch_total_params / 1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer_for_single_net(cfg.hyp,
                                                                                      net,
                                                                                      optimizer_state_dict, full_model=cfg.dataset.full_model)
    criterion = dt.utils.get_criterion(cfg.dataset.task)
    train_setup = dt.TrainingSetup(criterions=criterion,
                                   optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler)
    ####################################################

    ####################################################
    #        Train

    # cfg.work_dir = os.path.join(cfg.dataset.data_path, cfg.model.name)

    # os.makedirs(cfg.work_dir, exist_ok=True)

    if cfg.hyp.validate:
        meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(input_rule_name, k = 10*cfg.hyp.k, curr_dir=cfg.dataset.data_path)
        mata_class_derived_rule_bound_mappings_training, _, _ = validate_rule_main(loaders["train_rule_eval"], meta_class_mappings, 10*cfg.hyp.k, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, validate_rule_file_name=input_rule_name)
        mata_class_derived_rule_bound_mappings_valid, _, _ = validate_rule_main(loaders["valid_rule_eval"], meta_class_mappings, 10*cfg.hyp.k, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, validate_rule_file_name=input_rule_name)
        save_objs(meta_class_pred_boolean_mappings, os.path.join(cfg.dataset.data_path, "meta_class_pred_boolean_mappings"))

        save_objs(meta_class_rule_score_mappings, os.path.join(cfg.dataset.data_path, "meta_class_rule_score_mappings"))

        save_objs(mata_class_derived_rule_bound_mappings_training, os.path.join(cfg.dataset.data_path, "mata_class_derived_rule_bound_mappings_training"))

        save_objs(mata_class_derived_rule_bound_mappings_valid, os.path.join(cfg.dataset.data_path, "mata_class_derived_rule_bound_mappings_valid"))

        filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings = check_consistency_rule_bound_mappings(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid,topk=cfg.hyp.k)

        save_objs(filtered_meta_class_pred_boolean_mappings, os.path.join(cfg.dataset.data_path, "filtered_meta_class_pred_boolean_mappings"))
        save_objs(filtered_meta_class_rule_score_mappings, os.path.join(cfg.dataset.data_path, "filtered_meta_class_rule_score_mappings"))


        exit(1)
        # mata_class_derived_rule_bound_mappings_testing,_,_ = validate_rule_main(test_adapt_eval_loader, meta_class_id_mappings, k = args.topk, meta_class_pred_boolean_mappings=meta_class_pred_boolean_mappings, meta_class_rule_score_mappings=meta_class_rule_score_mappings)

        # save_objs(mata_class_derived_rule_bound_mappings_testing, os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_testing"))

        # valid_rule_eval

    eval_filtered_meta_class_pred_boolean_mappings, eval_filtered_meta_class_rule_score_mappings = None, None

    if "extensive_eval" in cfg.hyp and cfg.hyp.extensive_eval:
        mata_class_derived_rule_bound_mappings_training = load_objs(os.path.join(cfg.dataset.data_path, "mata_class_derived_rule_bound_mappings_training"))

        mata_class_derived_rule_bound_mappings_valid = load_objs(os.path.join(cfg.dataset.data_path, "mata_class_derived_rule_bound_mappings_valid"))
        
        eval_filtered_meta_class_pred_boolean_mappings, eval_filtered_meta_class_rule_score_mappings = check_consistency_rule_bound_mappings(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid,topk=cfg.hyp.eval_k)
        net.load_state_dict(torch.load(os.path.join(output_path, cfg.hyp.load_model_name)))
        curr_total_violation_count, curr_total_violation_loss = dt.eval_test_rule_violations(net, loaders["test_adapt_eval"], eval_filtered_meta_class_pred_boolean_mappings, eval_filtered_meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device)
        log.info(f"rule violation count: {curr_total_violation_count}")
        log.info(f"rule violation loss: {curr_total_violation_loss}")
        
    if cfg.dataset.train:
        log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
        highest_val_acc_so_far = -np.inf
        done = False
        epoch = start_epoch
        best_epoch = epoch

        while not done and epoch < cfg.hyp.epochs:
            # forward and backward pass for one whole epoch handeld inside dt.default_training_loop()
            loss = dt.default_training_loop(net, loaders["train"], train_setup, device)
            log.info(f"Training loss at epoch {epoch}: {loss}")

            # if the loss is nan, then stop the training
            if np.isnan(float(loss)):
                raise ValueError(f"{ic.format()} Loss is nan, exiting...")

            # TensorBoard writing
            # writer.add_scalar("Loss/loss", loss, epoch)
            # for i in range(len(optimizer.param_groups)):
            #     writer.add_scalar(f"Learning_rate/group{i}",
            #                       optimizer.param_groups[i]["lr"],
            #                       epoch)

            # evaluate the model periodically and at the final epoch
            if (epoch + 1) % cfg.hyp.val_period == 0 or epoch + 1 == cfg.hyp.epochs:
                test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                                    [loaders["test"], loaders["val"], loaders["train"]],
                                                                    cfg.dataset.task,
                                                                    device)
                log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
                log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
                log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

                # dt.utils.write_to_tb([train_stats["score"], val_stats["score"], test_stats["score"]],
                #                      [f"train_acc-{cfg.dataset.name}",
                #                       f"val_acc-{cfg.dataset.name}",
                #                       f"test_acc-{cfg.dataset.name}"],
                #                      epoch,
                #                      writer)

            if cfg.hyp.use_patience:
                val_stats, test_stats = dt.evaluate_model(net,
                                                        [loaders["val"], loaders["test"]],
                                                        cfg.dataset.task,
                                                        device)
                if val_stats["score"] > highest_val_acc_so_far:
                    best_epoch = epoch
                    highest_val_acc_so_far = val_stats["score"]
                    log.info(f"New best epoch, val score: {val_stats['score']}")
                    # save current model
                    state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
                    out_str = os.path.join(output_path, "model_best.pth")
                    log.info(f"Saving model to: {out_str}")
                    torch.save(state, out_str)

                if epoch - best_epoch > cfg.hyp.patience:
                    done = True
            epoch += 1
            # writer.flush()
            # writer.close()

        log.info("Running Final Evaluation...")
        checkpoint_path = os.path.join(output_path, "model_best.pth")
        net.load_state_dict(torch.load(checkpoint_path)["net"])
        test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                            [loaders["test"], loaders["val"], loaders["train"]],
                                                            cfg.dataset.task,
                                                            device)

        log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
        log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
        log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

        stats = OrderedDict([("dataset", cfg.dataset.name),
                            ("model_name", cfg.model.name),
                            ("run_id", cfg.run_id),
                            ("best_epoch", best_epoch),
                            ("routine", "from_scratch"),
                            ("test_stats", test_stats),
                            ("train_stats", train_stats),
                            ("val_stats", val_stats)])
        with open(os.path.join("stats.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
        log.info(json.dumps(stats, indent=4))
        ####################################################
        return stats
    else:
        assert cfg.hyp.tta_method in available_tta_methods

        normalizer = pickle.load(open(cfg.dataset.normalizer_path, 'rb'))

        # f1_low, f1_high = learning_statistics_for_bp_rule(cfg.hyp.train_batch_size, loaders["train"].dataset, normalizer)
        test_time_adaptation_main(cfg, log, start_epoch, net, loaders, train_setup,  output_path,device, optimizer, normalizer, meta_class_mappings, cache_path)




def test_time_adaptation_main_backup(cfg, log, start_epoch, net, loaders, train_setup,  output_path,device, optimizer, normalizer):
    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -np.inf
    done = False
    epoch = start_epoch
    best_epoch = epoch

    checkpoint_path = os.path.join(output_path, "model_best.pth")
    net.load_state_dict(torch.load(checkpoint_path)["net"])

    
    print(f'Normalizer loaded from {cfg.dataset.normalizer_path}')


    # dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
    test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                        [loaders["test"], loaders["val"], loaders["train"]],
                                                        cfg.dataset.task,
                                                        device)
    dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
    # dt.test_rule_violations(net, loaders["test"], device, normalizer)
    log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
    log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
    log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")
    while not done and epoch < cfg.hyp.epochs:
        # forward and backward pass for one whole epoch handeld inside dt.default_training_loop()
        loss = dt.default_test_time_adaptation_loop(net, loaders["test_adapt"], train_setup, device, normalizer)
        log.info(f"Training loss at epoch {epoch}: {loss}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard writing
        # writer.add_scalar("Loss/loss", loss, epoch)
        # for i in range(len(optimizer.param_groups)):
        #     writer.add_scalar(f"Learning_rate/group{i}",
        #                       optimizer.param_groups[i]["lr"],
        #                       epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.hyp.val_period == 0 or epoch + 1 == cfg.hyp.epochs:
            test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                                [loaders["test"], loaders["val"], loaders["train"]],
                                                                cfg.dataset.task,
                                                                device)
            dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
            # dt.test_rule_violations(net, loaders["test"], device, normalizer)
            log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
            log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
            log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")
            # 
            # tuple(st.bootstrap([f1s, ], np.median, method='percentile').confidence_interval)

            print()

            # dt.utils.write_to_tb([train_stats["score"], val_stats["score"], test_stats["score"]],
            #                      [f"train_acc-{cfg.dataset.name}",
            #                       f"val_acc-{cfg.dataset.name}",
            #                       f"test_acc-{cfg.dataset.name}"],
            #                      epoch,
            #                      writer)

        # if cfg.hyp.use_patience:
        #     val_stats, test_stats = dt.evaluate_model(net,
        #                                             [loaders["val"], loaders["test"]],
        #                                             cfg.dataset.task,
        #                                             device)
        #     if val_stats["score"] > highest_val_acc_so_far:
        #         best_epoch = epoch
        #         highest_val_acc_so_far = val_stats["score"]
        #         log.info(f"New best epoch, val score: {val_stats['score']}")
        #         # save current model
        #         state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
        #         out_str = os.path.join(output_path, "model_best.pth")
        #         log.info(f"Saving model to: {out_str}")
        #         torch.save(state, out_str)

        #     if epoch - best_epoch > cfg.hyp.patience:
        #         done = True
        epoch += 1
        # writer.flush()
        # writer.close()

    # log.info("Running Final Evaluation...")
    # checkpoint_path = os.path.join(output_path, "model_best.pth")
    # net.load_state_dict(torch.load(checkpoint_path)["net"])
    # test_stats, val_stats, train_stats = dt.evaluate_model(net,
    #                                                     [loaders["test"], loaders["val"], loaders["train"]],
    #                                                     cfg.dataset.task,
    #                                                     device)

    # log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
    # log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
    # log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

    # stats = OrderedDict([("dataset", cfg.dataset.name),
    #                     ("model_name", cfg.model.name),
    #                     ("run_id", cfg.run_id),
    #                     ("best_epoch", best_epoch),
    #                     ("routine", "from_scratch"),
    #                     ("test_stats", test_stats),
    #                     ("train_stats", train_stats),
    #                     ("val_stats", val_stats)])
    # with open(os.path.join("stats.json"), "w") as fp:
    #     json.dump(stats, fp, indent=4)
    # log.info(json.dumps(stats, indent=4))
    # ####################################################
    # return stats

def validate_rule_main(data_loader, meta_class_mappings, k = 20, meta_class_pred_boolean_mappings=None, meta_class_rule_score_mappings=None, validate_rule_file_name=input_rule_name):
    if meta_class_pred_boolean_mappings is None or meta_class_rule_score_mappings is None:
        meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(validate_rule_file_name, k = k)
    criterion = sigmoidF1()
    # eval_main(test_loader, model)
    
    mata_class_derived_rule_f1_score_ls_mappings = dict()
    mata_class_derived_rule_bound_mappings = dict()

    for meta_class in meta_class_pred_boolean_mappings:
        mata_class_derived_rule_f1_score_ls_mappings[meta_class] = []
        for idx in range(len(meta_class_pred_boolean_mappings[meta_class])):
            mata_class_derived_rule_f1_score_ls_mappings[meta_class].append([])

    # for item in tqdm(data_loader):
    #     image, target, path, df = item
    for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(data_loader, leave=False)):
        # inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                 inputs_cat if inputs_cat.nelement() != 0 else None

        validate_rules(df, targets, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, mata_class_derived_rule_f1_score_ls_mappings)

        
    for meta_class in mata_class_derived_rule_f1_score_ls_mappings:
        # mata_class_derived_rule_bounds_mappings[meta_class] = []
        mata_class_derived_rule_bound_mappings[meta_class]=[]
        for idx in range(len(meta_class_pred_boolean_mappings[meta_class])):
            f1_score_ls = mata_class_derived_rule_f1_score_ls_mappings[meta_class][idx]
            f1_lower_bound, f1_higher_bound = meta_class_rule_score_mappings[meta_class][idx]

            if len(f1_score_ls) > 1:
                f1_low_low, f1_low_high, f1_high_low, f1_high_high = calculate_confidence_interval(f1_score_ls)
            else:
                f1_low_low = -1
                f1_high_high = -1

            print("number of F1 scores::", len(f1_score_ls))

            print("original f1 score bounds::", f1_lower_bound, f1_higher_bound)

            print("f1 score bounds::", f1_low_low, f1_high_high)

            mata_class_derived_rule_bound_mappings[meta_class].append((f1_low_low, f1_high_high))

            print()

    return mata_class_derived_rule_bound_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings


def test_time_adaptation_main(cfg, log, start_epoch, net, loaders, train_setup,  output_path,device, optimizer, normalizer, meta_class_mappings, cache_path):

    
    if cfg.hyp.load_filtered_rules:
        meta_class_pred_boolean_mappings = load_objs(os.path.join(cfg.dataset.data_path, "filtered_meta_class_pred_boolean_mappings"))
        meta_class_rule_score_mappings = load_objs(os.path.join(cfg.dataset.data_path, "filtered_meta_class_rule_score_mappings"))
    else:
        meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(input_rule_name, k = cfg.hyp.k, curr_dir=cfg.dataset.data_path)

    rule_score_mappings = reconstruct_rule_for_all(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings)
    with open(os.path.join(output_path, "rule_score_mappings.json"), "w") as f:
        json.dump(rule_score_mappings, f, indent=4)

    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -np.inf
    done = False
    epoch = start_epoch
    best_epoch = epoch

    checkpoint_path = os.path.join(cache_path, "model_best.pth")
    net.load_state_dict(torch.load(checkpoint_path)["net"], strict=False)

    
    print(f'Normalizer loaded from {cfg.dataset.normalizer_path}')

    # if cfg.hyp.tta_method == 'rule':
    #     # eval_check_point_path = os.path.join(cache_path, "model_best.pth")
    #     perform_qualitative_studies_main(None, net, os.path.join(output_path, "before"), loaders["test"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device, cfg.hyp.test_batch_size)
    #     # model_path = os.path.join(output_path, "test_model_best.pth")
    #     # if os.path.exists(model_path):
    #     #     perform_qualitative_studies_main(model_path, net, os.path.join(output_path, "after"), loaders["test"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device, cfg.hyp.test_batch_size)
    # dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
    test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                        [loaders["test"], loaders["val"], loaders["train"]],
                                                        cfg.dataset.task,
                                                        device)
    # dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
    
    
    # perform_qualitative_studies_main(net, output_path, loaders["test"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device, cfg.hyp.test_batch_size)
    init_total_violation_count, init_total_violation_loss = dt.eval_test_rule_violations(net, loaders["test_adapt_eval"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device)
    # dt.test_rule_violations(net, loaders["test"], device, normalizer)
    log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
    log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
    log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")
    lowest_loss = np.inf
    lowest_violation_count = 0
    lowest_violation_loss = 0
    lowest_violation_count_reduced = 0
    lowest_violation_loss_reduced = 0
    lowest_training_stats = None
    lowest_val_stats = None
    lowest_test_stats = None



    while not done and epoch < cfg.hyp.epochs:
        # forward and backward pass for one whole epoch handeld inside dt.default_training_loop()
        loss = dt.default_test_time_adaptation_loop(cfg, net, loaders["test_adapt"], train_setup, device, normalizer, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
        log.info(f"Training loss at epoch {epoch}: {loss}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        if (epoch + 1) % cfg.hyp.val_period == 0 or epoch + 1 == cfg.hyp.epochs:
            test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                                [loaders["test"], loaders["val"], loaders["train"]],
                                                                cfg.dataset.task,
                                                                device)
            # dt.test_rule_violations(net, loaders["test_adapt_eval"], device, cfg.dataset.task, normalizer)
            curr_total_violation_count, curr_total_violation_loss = dt.eval_test_rule_violations(net, loaders["test_adapt_eval"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device)
            # dt.test_rule_violations(net, loaders["test"], device, normalizer)
            log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
            log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
            log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")
            violation_rule_count_reduction = (init_total_violation_count - curr_total_violation_count)/init_total_violation_count
            violation_rule_violation_loss_reduction = (init_total_violation_loss - curr_total_violation_loss)/init_total_violation_loss
            log.info(f"rule violation count reduced by {violation_rule_count_reduction}")
            log.info(f"rule violation loss reduced by {violation_rule_violation_loss_reduction}")
            torch.save(net.state_dict(), os.path.join(output_path, "test_model_" + str(epoch) + ".pth"))

            if loss < lowest_loss:
                lowest_violation_count = curr_total_violation_count
                lowest_violation_loss = curr_total_violation_loss
                lowest_violation_count_reduced = violation_rule_count_reduction
                lowest_violation_loss_reduced = violation_rule_violation_loss_reduction
                lowest_training_stats = train_stats
                lowest_val_stats = val_stats
                lowest_test_stats = test_stats
                log.info(f"with lowest test training loss Training accuracy: {json.dumps(lowest_training_stats, indent=4)}")
                log.info(f"with lowest test training loss Val accuracy: {json.dumps(lowest_val_stats, indent=4)}")
                log.info(f"with lowest test training loss Test accuracy: {json.dumps(lowest_test_stats, indent=4)}")
                log.info(f"with lowest test training loss rule violation count reduced by {lowest_violation_count_reduced}")
                log.info(f"with lowest test training loss rule violation loss reduced by {lowest_violation_loss_reduced}")
                log.info(f"with lowest test training loss rule total violation count:{lowest_violation_count}")
                log.info(f"with lowest test training loss rule total violation loss: {lowest_violation_loss}")
                lowest_loss = loss
                if os.path.exists(os.path.join(output_path, "test_model_best.pth")):
                    os.remove(os.path.join(output_path, "test_model_best.pth"))
                os.symlink(os.path.join(output_path, "test_model_" + str(epoch) + ".pth"), os.path.join(output_path, "test_model_best.pth"))
            # net, output_path, loaders["test"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device, cfg.hyp.test_batch_size
            
            #   
            # tuple(st.bootstrap([f1s, ], np.median, method='percentile').confidence_interval)

            print()

        epoch += 1
    # if cfg.hyp.tta_method == 'rule':
    #     eval_check_point_path = os.path.join(output_path, "test_model_best.pth")
    #     perform_qualitative_studies_main(eval_check_point_path, net, os.path.join(output_path, "after"), loaders["test"], meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, cfg.dataset.task, device, cfg.hyp.test_batch_size)
        
def perform_qualitative_studies_main(checkpoint_path, net, output_path, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, task, device, test_bz):
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path))
    
    os.makedirs(output_path, exist_ok=True)
    
    dt.perform_qualitative_studies(output_path, net, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, task, device, test_bz)
    
    
    
    

if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    
    main()
