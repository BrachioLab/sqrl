import argparse
from torchvision import models
import torch
from torchvision import transforms
from imagenet_dataset import MyImageNet, MyImageNet_test
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Subset
from tqdm import tqdm
import os
import torch.nn as nn
from imagenet_x import load_annotations
import pandas as pd
import random
import sys

import logging
from baseline_methods import memo_loss, entropy_classification_loss, conjugate_pl, robust_pl

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Tent.utils import calculate_confidence_interval
from rule_processing.process_rules import *
from rule_processing.sigmoidF1 import sigmoidF1
from rule_processing.dataset_for_sampling import Dataset_for_sampling
from baseline_methods import available_tta_methods
import pickle
import Tent.Tent as tent
import Norm.Norm as norm
from qualitative_study import perform_qualitative_studies


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = input.detach()
    return hook

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--epochs', type=int, default=10, help='used for resume')
    parser.add_argument('--output_rule_file_prefix', type=str, default="filtered_meta_class_", help='used for resume')
    parser.add_argument('--validate_rule_file_name', type=str, default="rule_f1_bounds_imagenetx.jsonl", help='used for resume')
    # output_rule_file_prefix
    # parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    # parser.add_argument('--lr', type=float, default=0.02, help='used for resume')
    parser.add_argument('--full_model',
        action='store_true',
        help='update full models')
    parser.add_argument('--batch_size', type=int, default=256, help='used for resume')
    parser.add_argument('--seed', type=int, default=0, help='used for resume')
    parser.add_argument('--test_batch_size', type=int, default=256, help='used for resume')
    parser.add_argument('--topk', type=int, default=-1, help='used for resume')
    parser.add_argument('--lr', type=float, default=0.002, help='used for resume')
    parser.add_argument('--cached_model_name', type=str, default="model_best", help='used for resume')
    # parser.add_argument('--model', type=str, default=0.02, help='used for resume', choices=["mlp", "dd"])
    parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--qualitative', action='store_true', help='use GPU')
    parser.add_argument('--validate', action='store_true', help='use GPU')
    parser.add_argument('--load_test_data', action='store_true', help='use GPU')
    parser.add_argument('--load_train_val_data', action='store_true', help='use GPU')
    parser.add_argument('--load_filtered_rules', action='store_true', help='use GPU')
    parser.add_argument('--qualitative_model_name', type=str, default="test_model_0", help='used for resume')
    parser.add_argument('--work_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--cache_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--data_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--tta_method', type=str, default="rule", help='used for resume')

    args = parser.parse_args()
    return args


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

def eval_main(val_loader, model):
    model.eval()
    logging.info("start evaluate model")
    val_loader.dataset.turn_off_aug=True
    # path2image_mappings = dict()
    # path2target_mappings = dict()
    # path2pred_mappings = dict()
    # val_loader.shuffle=False
    with torch.no_grad():
        total = 0
        correct = 0

        for item in tqdm(val_loader):
            image, target, path_ls, _ = item
            
            if type(image) is list:
                image = image[0]
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            # target = target.to(device, non_blocking=True)
            output = model(image)
            total += output.shape[0]
            pred_labels = torch.argmax(output, dim=-1)
            correct += torch.sum(pred_labels.view(-1) == target.view(-1)).item()
            # for k in range(len(path_ls)):
            #     path = path_ls[k]
            #     path2image_mappings[path] = image[k].cpu()
            #     path2target_mappings[path] = target[k].cpu()
            #     path2pred_mappings[path] = pred_labels[k].cpu()
            # del image, target, output, pred_labels
            # logging.info("validation correct::%f", correct)

        accuracy = correct*1.0/total
        logging.info("validation accuracy::%f", accuracy)

    model.train()

    return accuracy


def train_main(args, model, criterion, optimizer, train_loader, val_loader, save_dir):
    best_acc = 0

    for e in range(args.epochs):
        for item in tqdm(train_loader):
            image, target, path, _ = item
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            # target = target.to(device, non_blocking=True)
            output = model(image)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        valid_acc = eval_main(val_loader, model)

        torch.save(model.state_dict(), os.path.join(save_dir, "model_" + str(e)))

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best"))


def eval_test_rule_violations(model, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings):
    model.eval()
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        testloader.dataset.turn_off_aug=True
        for item in tqdm(testloader):
            image, target, path, df = item
            if type(image) is list:
                image = image[0]

            if torch.cuda.is_available():
                image = image.cuda()
            pred = model(image)
            violation_loss, violation_count = evaluate_rule_violations(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            # loss = compute_rule_loss(inputs_num, outputs, normalizer, criterion)
            total_violation_count += violation_count
            if violation_loss > 0:
                total_violation_loss += violation_loss.cpu().item()
            # if loss.cpu().item() > 0:
            # del image, pred, violation_loss

        logging.info("total violation count::%d", total_violation_count)
        logging.info("total_violation_loss::%f", total_violation_loss)

    model.train()




def eval_test_rule_violations2(dataset_for_sampling_loader, model, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, sampling_times = 1000):
    model.eval()
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        testloader.dataset.turn_off_aug=True
        all_meta_class_pred_rule_label_mappings = dict()
        batch_size = testloader.batch_size
        test_dataset_count = len(testloader.dataset)
        for item in tqdm(testloader):
            image, target, path, df = item
            if type(image) is list:
                image = image[0]

            if torch.cuda.is_available():
                image = image.cuda()
            pred = model(image)
            meta_class_pred_rule_label_mappings = obtain_rule_evaluations(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
            merge_meta_class_pred_rule_label_mappings(all_meta_class_pred_rule_label_mappings, meta_class_pred_rule_label_mappings)
            # loss = compute_rule_loss(inputs_num, outputs, normalizer, criterion)
            # total_violation_count += violation_count
            # if violation_loss > 0:
            #     total_violation_loss += violation_loss.cpu().item()
            # if loss.cpu().item() > 0:
            # del image, pred, violation_loss


        for sample_ids in tqdm(dataset_for_sampling_loader):
            rule_loss, rule_violations = post_eval_rule_f1_scores(sample_ids, all_meta_class_pred_rule_label_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings)
            total_violation_count += rule_violations
            total_violation_loss += rule_loss

        logging.info("total violation count::%d", total_violation_count)
        logging.info("total_violation_loss::%f", total_violation_loss)

    model.train()
    return total_violation_count, total_violation_loss


def filter_rules_by_symbolic_notations(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings):
    filtered_meta_class_pred_boolean_mappings = dict()
    filtered_meta_class_rule_score_mappings = dict()
    
    for meta_class in meta_class_pred_boolean_mappings:
        filtered_meta_class_pred_boolean_mappings[meta_class] = []
        filtered_meta_class_rule_score_mappings[meta_class] = []
        for idx in range(len(meta_class_pred_boolean_mappings[meta_class])):
            pred_boolean = meta_class_pred_boolean_mappings[meta_class][idx]
            keep = filter_rules_by_symbolic_conditions(pred_boolean)
            if not keep:
                continue
            else:
                filtered_meta_class_pred_boolean_mappings[meta_class].append(meta_class_pred_boolean_mappings[meta_class][idx])
                filtered_meta_class_rule_score_mappings[meta_class].append(meta_class_rule_score_mappings[meta_class][idx])
    
    return filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings

def test_time_adaptation_main(test_loader_for_model_eval, dataset_for_sampling_loader, args, save_dir, model, optimizer, test_loader, meta_class_mappings, output_prefix="filtered_meta_class_", k = 20, validate_rule_file_name = "rule_f1_bounds_imagenetx.jsonl"):
    if not args.load_filtered_rules:
        meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(validate_rule_file_name, k = k)
    else:
        meta_class_pred_boolean_mappings = load_objs(os.path.join(args.cache_dir, output_prefix + "pred_boolean_mappings"))
        meta_class_rule_score_mappings = load_objs(os.path.join(args.cache_dir, output_prefix + "rule_score_mappings"))
    # meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = filter_rules_by_symbolic_notations(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings)
        # select_top_k_rules_per_class(meta_claqss_pred_boolean_mappings, meta_class_rule_score_mappings, None, k=k)
    total_rule_count = sum([len(meta_class_pred_boolean_mappings[key]) for key in meta_class_pred_boolean_mappings])
    logging.info("total count of rules:%d", total_rule_count)
    criterion = sigmoidF1()
    # eval_main(test_loader, model)
    # eval_test_rule_violations(model, test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
    
    eval_test_rule_violations2(dataset_for_sampling_loader, model, test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)

    
    min_loss = np.inf
    min_loss_total_rule_violation_count = 0
    min_loss_total_rule_violation_loss = 0
    
    for e in range(args.epochs):
        if args.tta_method == "memo":
            test_loader.dataset.turn_off_aug=False
        
        total_loss = 0    
        
        for item in tqdm(test_loader):
            image, target, path, df = item
            if type(image) is not list:
                if torch.cuda.is_available():
                    image = image.cuda()
            else:
                image = [img.cuda() for img in image]
            if args.tta_method == "rule":
                pred = model(image)
                pred = torch.softmax(pred, dim=-1)
                loss = apply_rules_minibatch(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, criterion)
                del pred
            elif args.tta_method == "memo":
                pred0 = model(image[0])
                pred1 = model(image[1])
                pred2 = model(image[2])
                loss = memo_loss(pred0, pred1, pred2)
                del pred0, pred1, pred2
            if args.tta_method == "tent":
                pred = model(image)
                pred = torch.softmax(pred, dim=-1)
                loss = entropy_classification_loss(pred)
            if args.tta_method == "cpl":
                pred = model(image)
                loss = conjugate_pl(pred, num_classes=pred.shape[-1])
            if args.tta_method == "rpl":
                pred = model(image)
                loss = robust_pl(pred)
            if args.tta_method == "norm":
                pred = model(image)
                loss = 0
                del pred
            optimizer.zero_grad()
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
            
            
            del loss, image
        
        
        
        torch.save(model.state_dict(), os.path.join(save_dir, "test_model_" + str(e)))
        eval_main(test_loader_for_model_eval, model)
        # eval_test_rule_violations(model, test_eval_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
        total_violation_count, total_violation_loss = eval_test_rule_violations2(dataset_for_sampling_loader, model, test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
        if total_loss < min_loss:
            min_loss = total_loss
            min_loss_total_rule_violation_count = total_violation_count
            min_loss_total_rule_violation_loss = total_violation_loss
        logging.info("current min loss total violation count::%d", min_loss_total_rule_violation_count)
        logging.info("current min loss total_violation_loss::%f", min_loss_total_rule_violation_loss)

def validate_rule_main(data_loader, meta_class_mappings, k = 20, meta_class_pred_boolean_mappings=None, meta_class_rule_score_mappings=None, validate_rule_file_name="rule_f1_bounds.json"):
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

    print("start validating rules:")

    for item in tqdm(data_loader):
        image, target, path, df = item

        validate_rules(df, target, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, mata_class_derived_rule_f1_score_ls_mappings)

    print("start collecting statistics:")
    for meta_class in tqdm(mata_class_derived_rule_f1_score_ls_mappings):
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

            # print("number of F1 scores::", len(f1_score_ls))

            # print("original f1 score bounds::", f1_lower_bound, f1_higher_bound)

            # print("f1 score bounds::", f1_low_low, f1_high_high)

            mata_class_derived_rule_bound_mappings[meta_class].append((f1_low_low, f1_high_high))

            # print()

    return mata_class_derived_rule_bound_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings


def validate_rule_main2(dataset_for_sampling_loader, data_loader, meta_class_mappings, k = 20, meta_class_pred_boolean_mappings=None, meta_class_rule_score_mappings=None, validate_rule_file_name="rule_f1_bounds.json"):
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

    print("start validating rules:")
    all_meta_class_pred_rule_label_mappings = dict()
    for item in tqdm(data_loader):
        image, target, path, df = item

        # validate_rules(df, target, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, mata_class_derived_rule_f1_score_ls_mappings)
        meta_class_pred_rule_label_mappings = obtain_rule_evaluations(df, target, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
        merge_meta_class_pred_rule_label_mappings(all_meta_class_pred_rule_label_mappings, meta_class_pred_rule_label_mappings)
    
    # all_meta_class_rule_score_on_mb_mappings = dict()
    for sample_ids in tqdm(dataset_for_sampling_loader):
        # sub_df = full_df.iloc[sample_ids]
        rule_loss, rule_violations, meta_class_rule_score_on_mb_mappings = post_eval_rule_f1_scores(sample_ids, all_meta_class_pred_rule_label_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, return_details=True)
        merge_meta_class_f1_score_ls_mappings(mata_class_derived_rule_f1_score_ls_mappings, meta_class_rule_score_on_mb_mappings)
        
    print("start collecting statistics:")
    for meta_class in tqdm(mata_class_derived_rule_f1_score_ls_mappings):
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

            # print("number of F1 scores::", len(f1_score_ls))

            # print("original f1 score bounds::", f1_lower_bound, f1_higher_bound)

            # print("f1 score bounds::", f1_low_low, f1_high_high)

            mata_class_derived_rule_bound_mappings[meta_class].append((f1_low_low, f1_high_high))

            # print()

    return mata_class_derived_rule_bound_mappings, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings





def get_meta_class_image_class_mappings():
    annotations1 = load_annotations(partition="train")
    annotations2 = load_annotations(partition="val")
    full_annotations = pd.concat([annotations1, annotations2])

    df_class_mappings = full_annotations.groupby(by=["metaclass"])["class"].apply(list)

    class_id_meta_class_mappings = dict()

    for idx in range(len(df_class_mappings)):
        meta_class = df_class_mappings.index[idx]
        class_ls = df_class_mappings[meta_class]
        for class_id in class_ls:
            class_id_meta_class_mappings[class_id] = meta_class

    unique_meta_class_ls = list(annotations1["metaclass"].unique())
    unique_meta_class_ls.sort()

    meta_class_id_mappings = dict()
    for idx in range(len(unique_meta_class_ls)):
        meta_class = unique_meta_class_ls[idx]
        meta_class_id_mappings[meta_class] = idx


    return class_id_meta_class_mappings, meta_class_id_mappings


def get_existing_test_samples(args, test_dataset, class_id_meta_class_mappings, pre_process=None):
    print("Loading Annotations")
    annotations = load_annotations(partition="val")
    print("Loaded")
    file_name_set = set(annotations["file_name"])
    existing_id_ls = []
    for idx in range(len(test_dataset)):
        path, target = test_dataset.samples[idx]

        file_name = path.split("/")[-1]

        if file_name in file_name_set and target in class_id_meta_class_mappings:
            existing_id_ls.append(idx)

    print("Init")
    sample_ls, target_ls, df_ls, path_ls = test_dataset.init_samples(existing_id_ls)
    # test_dataset = Subset(test_dataset, existing_id_ls)
    print("Init finished")
    augment = False
    if args.tta_method == "memo":
        augment = True
    test_dataset = MyImageNet_test(sample_ls, target_ls, df_ls, path_ls, test_dataset.transform, test_dataset.target_transform, augment=augment,preprocess=pre_process, new_path="/data6/wuyinjun/imagenet/imagent-c/zoom_blur/5/")

    return test_dataset

def get_existing_training_samples_in_imagenet_x(dataset, full_set, class_id_meta_class_mappings):
    annotations = load_annotations(partition="train")
    file_name_set = set(annotations["file_name"])
    existing_id_ls = []
    for idx in tqdm(range(len(dataset.indices))):

        _, target,path,_ = dataset[idx]

        file_name = path.split("/")[-1]

        if file_name in file_name_set and target in class_id_meta_class_mappings:
            existing_id_ls.append(dataset.dataset.indices[dataset.indices[idx]])


    sample_ls, target_ls, df_ls, path_ls = full_set.init_samples(torch.tensor(existing_id_ls).tolist(), split='train')
    # test_dataset = Subset(test_dataset, existing_id_ls)
    dataset = MyImageNet_test(sample_ls, target_ls, df_ls, path_ls, full_set.transform, full_set.target_transform)

    return dataset

def get_existing_train_samples(test_dataset, class_id_meta_class_mappings):

    existing_id_ls = []
    for idx in range(len(test_dataset)):
        path, target = test_dataset.samples[idx]

        file_name = path.split("/")[-1]

        if target in class_id_meta_class_mappings:
            existing_id_ls.append(idx)

    test_dataset = Subset(test_dataset, existing_id_ls)

    return test_dataset

class model_wrapper():
    def __init__(self):
        self.output_embedding = dict()

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.output_embedding[layer_name] = input[0].cpu().detach()
        return hook


def pre_compute_embeddings(model, train_loader):

    mw = model_wrapper()
    model.fc.register_forward_hook(mw.forward_hook('feats'))
    
    output_embedding_ls = []
    target_ls = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(train_loader):
            image, target, path, _ = item
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()

            pred = model(image)
            output_embedding = mw.output_embedding["feats"]
            output_embedding_ls.append(output_embedding)
            target_ls.append(target)
    output_embedding_tensor = torch.cat(output_embedding_ls)
    target_tensor = torch.cat(target_ls)

    return output_embedding_tensor, target_tensor

def training_using_embeddings(args, model, embedding, targets, ciriterion, test_loader):
    optimizer = torch.optim.SGD(model.fc.parameters(), lr = args.lr)

    for e in tqdm(range(args.epochs)):
        rand_ids = torch.randperm(embedding.shape[0])
        for start_idx in range(0, embedding.shape[0], args.batch_size):
            
            end_idx = start_idx + args.batch_size
            if end_idx > embedding.shape[0]:
                end_idx = embedding.shape[0]

            indices = rand_ids[start_idx: end_idx]
            embedding_mb = embedding[indices]
            target_mb = targets[indices]
            
            output = model.fc(embedding_mb)
            loss = ciriterion(output, target_mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_main(test_loader, model)


def iterate_test_dataset(test_loader):
    all_target_ls = []

    for item in tqdm(test_loader):
        image, target, path, df = item
        all_target_ls.append(target)

    target_tensor = torch.cat(all_target_ls)

    print()


def save_objs(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def load_objs(file):
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_model_embeddings(model, data_loader):
    model.eval()
    logging.info("start evaluate model")
    data_loader.dataset.turn_off_aug=True
    # path2image_mappings = dict()
    # path2target_mappings = dict()
    # path2pred_mappings = dict()
    # val_loader.shuffle=False
    with torch.no_grad():
        total = 0
        correct = 0

        data_embedding_ls = []

        for item in tqdm(data_loader):
            image, target, path_ls, _ = item
            
            # for path in path_ls:
            #     if path == "/data6/wuyinjun/imagenet/ILSVRC/Data/CLS-LOC/val/n03075370/ILSVRC2012_val_00037402.JPEG":
            #         print()
            
            if type(image) is list:
                image = image[0]
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            # target = target.to(device, non_blocking=True)
            output = model(image)
            data_embedding = activation['fc']
            data_embedding_ls.append(data_embedding)
        
        return torch.cat(data_embedding_ls)


def main(args):
    print(args)

    set_random_seed(args.seed)

    log_file_name = os.path.join(args.work_dir, "log.txt")

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    logging.basicConfig(filename=log_file_name,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    logging.info("start logging")

    # model = models.resnet34(pretrained=True)

    # model = models.resnet18(pretrained=True)
    model = models.resnet34(pretrained=True)

    # model = models.resnet18(pretrained=True)
    

    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    if not args.tta_method == "memo":
        test_transform = transform
        pre_process = None
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ])
        pre_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    class_id_meta_class_mappings, meta_class_id_mappings = get_meta_class_image_class_mappings()
    model.fc = nn.Linear(512, 17)# len(meta_class_id_mappings)
    # model.fc.register_forward_hook(get_activation('fc'))
    



    all_train_full_dataset = MyImageNet(os.path.join(args.data_dir,"Data/CLS-LOC/train/"), split='train', transform=transform, class_id_meta_class_mappings = class_id_meta_class_mappings, meta_class_id_mappings = meta_class_id_mappings)
    if torch.cuda.is_available():
        model = model.cuda()
    all_train_dataset = get_existing_train_samples(all_train_full_dataset, class_id_meta_class_mappings)
    # all_sample_ids = list(range(len(train_dataset)))
    save_dir = os.path.join(args.work_dir, "train/")
    os.makedirs(save_dir, exist_ok=True)
    if not args.load_train_val_data:
        rand_sample_ids = torch.randperm(len(all_train_dataset))
        train_count = int(len(all_train_dataset)*0.9)
        train_sample_ids = rand_sample_ids[0:train_count]
        valid_sample_ids = rand_sample_ids[train_count:]
        save_objs(train_sample_ids, os.path.join(save_dir, "train_sample_ids"))
        save_objs(valid_sample_ids, os.path.join(save_dir, "valid_sample_ids"))
    else:
        print("Loading objects")
        train_sample_ids = load_objs(os.path.join(args.cache_dir, "train_sample_ids"))
        valid_sample_ids = load_objs(os.path.join(args.cache_dir, "valid_sample_ids"))
        print("Loaded")

    train_dataset = Subset(all_train_dataset, train_sample_ids)
    valid_dataset = Subset(all_train_dataset, valid_sample_ids)
    valid_loader = DataLoader(valid_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=False)
    



    # validate_rule_main(valid_loader, meta_class_id_mappings, k = 20)
    if args.validate:
        if not args.load_train_val_data:
            existing_training_dataset =  get_existing_training_samples_in_imagenet_x(train_dataset, all_train_full_dataset, class_id_meta_class_mappings)
            existing_valid_dataset =  get_existing_training_samples_in_imagenet_x(valid_dataset, all_train_full_dataset, class_id_meta_class_mappings)
            save_objs(existing_training_dataset, os.path.join(save_dir, "existing_training_dataset"))
            save_objs(existing_valid_dataset, os.path.join(save_dir, "existing_valid_dataset"))
        else:
            print("loading data")
            existing_training_dataset = load_objs(os.path.join(args.cache_dir, "existing_training_dataset"))
            existing_valid_dataset = load_objs(os.path.join(args.cache_dir, "existing_valid_dataset"))
    if not args.load_test_data:
        test_dataset = MyImageNet(os.path.join(args.data_dir, "Data/CLS-LOC/val/"), split='val', use_annotation=True, transform=test_transform, class_id_meta_class_mappings = class_id_meta_class_mappings, meta_class_id_mappings = meta_class_id_mappings)
        print("Getting test samples")
        test_dataset = get_existing_test_samples(args, test_dataset, class_id_meta_class_mappings, pre_process)
        print("Saving test dataset")
        save_objs(test_dataset, os.path.join(save_dir, "test_data"))
        print("Saved")
    else:
        test_dataset = load_objs(os.path.join(args.cache_dir, "test_data"))

        if args.tta_method == "memo":
            test_dataset.transform = test_transform
            test_dataset.augment = True
            test_dataset.preprocess = pre_process
            test_dataset2 = load_objs(os.path.join(args.cache_dir, "test_data"))
    
    
    train_loader = DataLoader(train_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=True)
    if args.tta_method == "memo":
        test_loader = DataLoader(test_dataset2, collate_fn = MyImageNet.collate_fn,batch_size=args.test_batch_size,shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.test_batch_size,shuffle=True)
    # test_loader2 = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.test_batch_size,shuffle=False)
    test_time_adapt_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=True)

    # sampler = RandomSampler(test_dataset, replacement=True, num_samples=len(test_dataset)*5)
    # test_adapt_eval_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.test_batch_size)

    dataset_for_sampling = Dataset_for_sampling(len(test_dataset))
    sampler = RandomSampler(dataset_for_sampling, replacement=True, num_samples=200*args.test_batch_size)
    dataset_for_sampling_loader = DataLoader(dataset_for_sampling, collate_fn = Dataset_for_sampling.collate_fn,sampler=sampler, batch_size=args.test_batch_size)

    print("start validating")

    if args.validate:
        
        # sampler = RandomSampler(existing_training_dataset, replacement=True, num_samples=100)
        # existing_training_loader = DataLoader(existing_training_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.batch_size)

        # sampler = RandomSampler(existing_valid_dataset, replacement=True, num_samples=100)
        # existing_valid_loader = DataLoader(existing_valid_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.batch_size)
        existing_training_loader = DataLoader(existing_training_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=False)
        existing_valid_loader = DataLoader(existing_valid_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=False)
        
        dataset_for_sampling_training = Dataset_for_sampling(len(existing_training_dataset))
        sampler = RandomSampler(dataset_for_sampling_training, replacement=True, num_samples=200*args.batch_size)
        dataset_for_sampling_loader_training = DataLoader(dataset_for_sampling_training, collate_fn = Dataset_for_sampling.collate_fn,sampler=sampler, batch_size=args.batch_size)
        
        
        mata_class_derived_rule_bound_mappings_training, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = validate_rule_main2(dataset_for_sampling_loader_training, existing_training_loader, meta_class_id_mappings, k = args.topk, validate_rule_file_name=args.validate_rule_file_name)

        save_objs(meta_class_pred_boolean_mappings, os.path.join(save_dir, "meta_class_pred_boolean_mappings"))

        save_objs(meta_class_rule_score_mappings, os.path.join(save_dir, "meta_class_rule_score_mappings"))

        save_objs(mata_class_derived_rule_bound_mappings_training, os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_training"))

        dataset_for_sampling_valid = Dataset_for_sampling(len(existing_valid_dataset))
        sampler = RandomSampler(dataset_for_sampling_valid, replacement=True, num_samples=200*args.batch_size)
        dataset_for_sampling_loader_valid = DataLoader(dataset_for_sampling_valid, collate_fn = Dataset_for_sampling.collate_fn,sampler=sampler, batch_size=args.batch_size)

        mata_class_derived_rule_bound_mappings_valid,_,_ = validate_rule_main2(dataset_for_sampling_loader_valid, existing_valid_loader, meta_class_id_mappings, k = args.topk, meta_class_pred_boolean_mappings=meta_class_pred_boolean_mappings, meta_class_rule_score_mappings=meta_class_rule_score_mappings)

        save_objs(mata_class_derived_rule_bound_mappings_valid, os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_valid"))

        mata_class_derived_rule_bound_mappings_testing,_,_ = validate_rule_main2(dataset_for_sampling_loader, test_loader, meta_class_id_mappings, k = args.topk, meta_class_pred_boolean_mappings=meta_class_pred_boolean_mappings, meta_class_rule_score_mappings=meta_class_rule_score_mappings)

        save_objs(mata_class_derived_rule_bound_mappings_testing, os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_testing"))

        exit(1)
    # iterate_test_dataset(train_loader)
    if args.qualitative:
        # model.load_state_dict(torch.load(os.path.join(args.cache_dir, args.cached_model_name)), strict=False)
        if not args.load_filtered_rules:
            meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(args.validate_rule_file_name, k = args.topk)
        else:
            meta_class_pred_boolean_mappings = load_objs(os.path.join(args.cache_dir, args.output_rule_file_prefix + "pred_boolean_mappings"))
            meta_class_rule_score_mappings = load_objs(os.path.join(args.cache_dir, args.output_rule_file_prefix + "rule_score_mappings"))
            
        rule_score_mappings = reconstruct_rule_for_all(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings)
        with open(os.path.join(args.cache_dir, "rule_score_mappings.json"), "w") as f:
            json.dump(rule_score_mappings, f, indent=4)
        # perform_qualitative_studies_main(os.path.join(args.cache_dir, args.cached_model_name), model, args.cached_model_name, os.path.join(args.cache_dir, "before"), test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_id_mappings, args.test_batch_size)    
        perform_qualitative_studies_main(os.path.join(args.cache_dir, args.qualitative_model_name), model, args.qualitative_model_name, os.path.join(args.cache_dir, "after"), test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_id_mappings, args.test_batch_size)    
        exit(1)
    if args.train:
        
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(args.cache_dir, args.cached_model_name)):
            logging.info("loading prev model")
            model.load_state_dict(torch.load(os.path.join(args.cache_dir, args.cached_model_name)), strict=False)
        criterion = nn.CrossEntropyLoss()
        # for param in model.parameters():
        #     param.train = False

        # model.fc.train = True


        # print("start transform features")
        # embedding, targets = pre_compute_embeddings(model, train_loader)
        # print("train using embeddings")
        # training_using_embeddings(args, model, embedding, targets, criterion, test_loader)



        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
        eval_main(test_loader, model)

        eval_main(valid_loader, model)
        train_main(args, model, criterion, optimizer, train_loader, valid_loader, save_dir)
        print("test performance")
        eval_main(test_loader, model)

    else:

        assert args.tta_method in available_tta_methods
        save_dir = args.work_dir

        model.load_state_dict(torch.load(os.path.join(args.cache_dir, args.cached_model_name)), strict=False)
        # model.load_state_dict(torch.load(os.path.join(save_dir, "test_model_1")), strict=False)
        # eval_main(test_time_adapt_loader, model)
        # eval_main(test_loader, model)
        if args.tta_method == "norm":
            model = norm.Norm(model)
            params = model.parameters()
        else:
            if not args.full_model:
                model = tent.configure_model(model)
                params, param_names = tent.collect_params(model)
            else:
                model = tent.configure_model2(model)
                params = model.parameters()
        
        all_params = [{"params": params}]
        optimizer = torch.optim.SGD(all_params, lr = args.lr)
        # new_path = test_loader.dataset.new_path

        # test_loader.dataset.new_path = None

        # eval_main(test_loader, model)
        # # eval_main(test_time_adapt_loader, model)
        # test_loader.dataset.new_path = new_path

        eval_main(test_loader, model)

        
        
        test_time_adaptation_main(test_loader, dataset_for_sampling_loader, args, save_dir, model, optimizer, test_time_adapt_loader, meta_class_id_mappings, output_prefix=args.output_rule_file_prefix, k =args.topk, validate_rule_file_name=args.validate_rule_file_name)

def perform_qualitative_studies_main(checkpoint_path, net, qualitative_model_name, output_path, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, test_bz):
    # checkpoint_path = 
    net.load_state_dict(torch.load(checkpoint_path), strict=False)
    net = tent.configure_model(net)
    os.makedirs(output_path, exist_ok=True)
    perform_qualitative_studies(output_path, net, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, test_bz)

    

if __name__ == '__main__':
    args = parse_args()
    main(args)