
from train_and_test_time_adaptation import parse_args, get_meta_class_image_class_mappings, get_existing_train_samples, save_objs, load_objs, get_existing_training_samples_in_imagenet_x, get_existing_test_samples
import os
import logging

from torchvision import models
import torch
from torchvision import transforms
import os



import sys

import logging

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rule_processing.process_rules import *

def main(args):

    log_file_name = os.path.join(args.work_dir, "log.txt")

    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    logging.basicConfig(filename=log_file_name,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    logging.info("start logging")

    model = models.resnet34(pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # class_id_meta_class_mappings, meta_class_id_mappings = get_meta_class_image_class_mappings()
    # model.fc.out_features = len(meta_class_id_mappings)
    # all_train_full_dataset = MyImageNet(os.path.join(args.data_dir,"Data/CLS-LOC/train/"), split='train', transform=transform, class_id_meta_class_mappings = class_id_meta_class_mappings, meta_class_id_mappings = meta_class_id_mappings)

    # all_train_dataset = get_existing_train_samples(all_train_full_dataset, class_id_meta_class_mappings)
    # all_sample_ids = list(range(len(train_dataset)))
    save_dir = os.path.join(args.work_dir, "train/")
    # if not args.load_train_val_data:
    #     rand_sample_ids = torch.randperm(len(all_train_dataset))
    #     train_count = int(len(all_train_dataset)*0.9)
    #     train_sample_ids = rand_sample_ids[0:train_count]
    #     valid_sample_ids = rand_sample_ids[train_count:]
    #     save_objs(train_sample_ids, os.path.join(save_dir, "train_sample_ids"))
    #     save_objs(valid_sample_ids, os.path.join(save_dir, "valid_sample_ids"))
    # else:
    #     train_sample_ids = load_objs(os.path.join(save_dir, "train_sample_ids"))
    #     valid_sample_ids = load_objs(os.path.join(save_dir, "valid_sample_ids"))
    # train_dataset = Subset(all_train_dataset, train_sample_ids)
    # valid_dataset = Subset(all_train_dataset, valid_sample_ids)
    # valid_loader = DataLoader(valid_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=False)
    
    # # validate_rule_main(valid_loader, meta_class_id_mappings, k = 20)
    # if args.validate:
    #     if not args.load_train_val_data:
    #         existing_training_dataset =  get_existing_training_samples_in_imagenet_x(train_dataset, all_train_full_dataset, class_id_meta_class_mappings)
    #         existing_valid_dataset =  get_existing_training_samples_in_imagenet_x(valid_dataset, all_train_full_dataset, class_id_meta_class_mappings)
    #         save_objs(existing_training_dataset, os.path.join(save_dir, "existing_training_dataset"))
    #         save_objs(existing_valid_dataset, os.path.join(save_dir, "existing_valid_dataset"))
    #     else:
    #         print("loading data")
    #         existing_training_dataset = load_objs(os.path.join(save_dir, "existing_training_dataset"))
    #         existing_valid_dataset = load_objs(os.path.join(save_dir, "existing_valid_dataset"))
    # if not args.load_test_data:
    #     test_dataset = MyImageNet(os.path.join(args.data_dir, "Data/CLS-LOC/val/"), split='val', use_annotation=True, transform=transform, class_id_meta_class_mappings = class_id_meta_class_mappings, meta_class_id_mappings = meta_class_id_mappings)

    #     test_dataset = get_existing_test_samples(test_dataset, class_id_meta_class_mappings)

    #     save_objs(test_dataset, os.path.join(save_dir, "test_data"))
    # else:
    #     test_dataset = load_objs(os.path.join(save_dir, "test_data"))
    
    
    # train_loader = DataLoader(train_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=True)
    # test_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=False)
    # test_time_adapt_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,batch_size=args.batch_size,shuffle=True)

    # sampler = RandomSampler(test_dataset, replacement=True, num_samples=len(test_dataset)*2)
    # test_adapt_eval_loader = DataLoader(test_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.batch_size)

    # print("start validating")

    # # if args.validate:
    # sampler = RandomSampler(existing_training_dataset, replacement=True, num_samples=len(existing_training_dataset)*2)
    # existing_training_loader = DataLoader(existing_training_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.batch_size)

    # sampler = RandomSampler(existing_valid_dataset, replacement=True, num_samples=len(existing_valid_dataset)*2)
    # existing_valid_loader = DataLoader(existing_valid_dataset, collate_fn = MyImageNet.collate_fn,sampler=sampler, batch_size=args.batch_size)

    # mata_class_derived_rule_bound_mappings_training, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = validate_rule_main(existing_training_loader, meta_class_id_mappings, k = args.topk)

    meta_class_pred_boolean_mappings = load_objs(os.path.join(save_dir, "meta_class_pred_boolean_mappings"))

    meta_class_rule_score_mappings = load_objs(os.path.join(save_dir, "meta_class_rule_score_mappings"))

    mata_class_derived_rule_bound_mappings_training = load_objs(os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_training"))

    # mata_class_derived_rule_bound_mappings_valid,_,_ = validate_rule_main(existing_valid_loader, meta_class_id_mappings, k = args.topk, meta_class_pred_boolean_mappings=meta_class_pred_boolean_mappings, meta_class_rule_score_mappings=meta_class_rule_score_mappings)

    mata_class_derived_rule_bound_mappings_valid = load_objs(os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_valid"))


    filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings = check_consistency_rule_bound_mappings_imagenet(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid, topk=args.topk)

    save_objs(filtered_meta_class_pred_boolean_mappings, os.path.join(save_dir, "filtered_meta_class_pred_boolean_mappings"))
    save_objs(filtered_meta_class_rule_score_mappings, os.path.join(save_dir, "filtered_meta_class_rule_score_mappings"))
    # mata_class_derived_rule_bound_mappings_testing,_,_ = validate_rule_main(test_adapt_eval_loader, meta_class_id_mappings, k = args.topk, meta_class_pred_boolean_mappings=meta_class_pred_boolean_mappings, meta_class_rule_score_mappings=meta_class_rule_score_mappings)

    # save_objs(mata_class_derived_rule_bound_mappings_testing, os.path.join(save_dir, "mata_class_derived_rule_bound_mappings_testing"))


# def check_consistency_rule_bound_mappings(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, mata_class_derived_rule_bound_mappings_training, mata_class_derived_rule_bound_mappings_valid, topk=20):
#     filtered_meta_class_pred_boolean_mappings = dict()
#     filtered_meta_class_rule_score_mappings = dict()
#     filtered_meta_class_rule_overlap_mappings = dict()
    
#     for meta_class in mata_class_derived_rule_bound_mappings_training:
#         rule_bound_ls_training = mata_class_derived_rule_bound_mappings_training[meta_class]
#         rule_bound_ls_valid = mata_class_derived_rule_bound_mappings_valid[meta_class]
#         filtered_meta_class_pred_boolean_mappings[meta_class] = []
#         filtered_meta_class_rule_score_mappings[meta_class] = []
#         filtered_meta_class_rule_overlap_mappings[meta_class] = []
#         curr_meta_class_pred_boolean_ls = meta_class_pred_boolean_mappings[meta_class]
#         for idx in range(len(rule_bound_ls_training)):
#             rule_bound_training = rule_bound_ls_training[idx]
#             rule_bound_valid = rule_bound_ls_valid[idx]
#             if rule_bound_valid[0] < 0 or rule_bound_valid[1] < 0 or rule_bound_training[0] < 0 or rule_bound_training[1] < 0:
#                 continue
#             min_lb = min(rule_bound_training[0], rule_bound_valid[0])
#             max_lb = max(rule_bound_training[0], rule_bound_valid[0])
#             min_hb = min(rule_bound_training[1], rule_bound_valid[1])
#             max_hb = max(rule_bound_training[1], rule_bound_valid[1])


#             if min_hb < max_lb:
#                 overlap = 0
#             elif max_hb == min_lb:
#                 overlap = 1
#             else:
#                 overlap = (min_hb - max_lb)/(max_hb - min_lb)

#             if overlap > 0.8:# and rule_bound_training[1] - rule_bound_training[0] < 1 and rule_bound_training[1] - rule_bound_training[0] > 0:
#             # if overlap > 0.8 and rule_bound_training[0] > 0:
#                 # print(rule_bound_training, rule_bound_valid)
#                 filtered_meta_class_pred_boolean_mappings[meta_class].append(curr_meta_class_pred_boolean_ls[idx])
#                 filtered_meta_class_rule_score_mappings[meta_class].append((rule_bound_training[0], rule_bound_training[1]))
#                 filtered_meta_class_rule_overlap_mappings[meta_class].append(overlap)
#         overlap_score_ls = torch.tensor(filtered_meta_class_rule_overlap_mappings[meta_class])
#         print("number of overlap rules for metaclass %s:%d"%(meta_class, len(overlap_score_ls)))
#         sorted_overlap_sore_ls, sorted_ids = torch.sort(overlap_score_ls, descending=True)
#         selected_sorted_ids = sorted_ids[0:topk].tolist()
#         selected_pred_boolean_mappings = [filtered_meta_class_pred_boolean_mappings[meta_class][k] for k in selected_sorted_ids]
#         filtered_meta_class_pred_boolean_mappings[meta_class] = selected_pred_boolean_mappings

#         selected_rule_score_mappings = [filtered_meta_class_rule_score_mappings[meta_class][k] for k in selected_sorted_ids]
#         filtered_meta_class_rule_score_mappings[meta_class] = selected_rule_score_mappings

#         print("number of rules for metaclass %s:%d"%(meta_class, len(filtered_meta_class_pred_boolean_mappings[meta_class])))

#     return filtered_meta_class_pred_boolean_mappings, filtered_meta_class_rule_score_mappings

if __name__ == '__main__':

    args = parse_args()
    main(args)