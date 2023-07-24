import argparse
# from pathlib import Path
import shutil
import os
import logging
import sys
from tqdm import tqdm
import pickle
# curr_path = Path.cwd()
# project_dir = Path.cwd().parent

# sys.path.append('..')
# sys.path.append(os.path.join(curr_path, "finBERT"))
# print(project_dir)
# print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from interval3 import Interval, IntervalSet
from baseline_methods import memo_loss, entropy_classification_loss, conjugate_pl, robust_pl
from sklearn.metrics import f1_score

from rule_processing.sigmoidF1 import sigmoidF1
from rule_processing.process_rules import apply_rules_single_per_sample, apply_rules_single_per_sample_evaluation, apply_rules_single_per_sample_evaluation_full
from rule_processing.process_rules import apply_rules_minibatch, obtain_rule_evaluations, merge_meta_class_pred_rule_label_mappings, post_eval_rule_f1_scores
from rule_processing.dataset_for_sampling import Dataset_for_sampling

from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report

from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset


from finbert.finbert import *
import finbert.utils as tools

import csv
import ast

import Tent.Tent as tent
import Norm.Norm as norm

import torch
import  torch.nn.functional as F

class custom_dataset(TensorDataset):
    # tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors):
        super().__init__(*tensors)
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # self.tensors = tensors

    def __getitem__(self, index):
        return index, tuple(tensor[index] for tensor in self.tensors)

    # def __len__(self):
    #     return self.tensors[0].size(0)

    # def __init__(self, dataset) -> None:
    #     # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    #     # self.tensors = tensors
    #     self.dataset = dataset

    # def __getitem__(self, index):
    #     return index, self.dataset[index]

    # def __len__(self):
    #     return self.tensors[0].size(0)

    @staticmethod
    def collate_fn(data):
        index_ls = torch.tensor([data[i][0] for i in range(len(data))])
        res_ls = []
        for k in range(len(data[0][1])):
            res_ls.append(torch.stack([data[i][1][k] for i in range(len(data))]))
        return index_ls, res_ls
    
def get_loader(finbert, examples, phase):
    """
    Creates a data loader object for a dataset.
    Parameters
    ----------
    examples: list
        The list of InputExample's.
    phase: 'train' or 'eval'
        Determines whether to use random sampling or sequential sampling depending on the phase.
    Returns
    -------
    dataloader: DataLoader
        The data loader object.
    """

    features = convert_examples_to_features(examples, finbert.label_list,
                                            finbert.config.max_seq_length,
                                            finbert.tokenizer,
                                            finbert.config.output_mode)

    # Log the necessasry information
    logger.info("***** Loading data *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", finbert.config.train_batch_size)
    logger.info("  Num steps = %d", finbert.num_train_optimization_steps)

    # Load the data, make it into TensorDataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_tokens = [f.tokens for f in features]

    if finbert.config.output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif finbert.config.output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    try:
        all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
    except:
        all_agree_ids = torch.tensor([0.0 for f in features], dtype=torch.long)

    dataset = custom_dataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)

    # dataset = custom_dataset(data)

    # Distributed, if necessary
    if phase == 'train':
        my_sampler = RandomSampler(dataset)
    elif phase == 'eval':
        my_sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=my_sampler, batch_size=finbert.config.train_batch_size, collate_fn=custom_dataset.collate_fn)
    return dataloader, all_tokens

def evaluate_rule_violation_count(finbert, test_loader, model, test_df, meta_class_rule):
    iterator = tqdm(enumerate(test_loader), desc="test violation evaluation", total=len(test_loader))
    total_violation_count = 0
    # for sample_ids, batch in tqdm(test_loader_for_training):
    for _, (sample_ids, batch) in iterator:
        batch = tuple(t.to(finbert.device) for t in batch)

        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

        curr_test_df = test_df.iloc[sample_ids.numpy()]

        logits = model(input_ids, attention_mask, token_type_ids)[0]
        
        # if args.tta_method == "rule":
        total_violation_count += apply_rules_single_per_sample_evaluation(curr_test_df, logits, meta_class_rule)

    # logger.info("total rule violations::")    
    # logger.info(str(total_violation_count))
    return total_violation_count

def evaluate_rule_violation_count_compare(finbert, test_loader, test_res1, test_res2, test_df, meta_class_rule):
    iterator = tqdm(enumerate(test_loader), desc="test violation evaluation", total=len(test_loader))
    total_violation_count = 0
    # for sample_ids, batch in tqdm(test_loader_for_training):
    for _, (sample_ids, batch) in iterator:
        batch = tuple(t.to(finbert.device) for t in batch)

        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

        curr_test_df = test_df.iloc[sample_ids.numpy()]

        # logits1 = model1(input_ids, attention_mask, token_type_ids)[0]
        # logits2 = model2(input_ids, attention_mask, token_type_ids)[0]
        pred_label1 = torch.tensor(list(test_res1.iloc[sample_ids.numpy()]["prediction"])).view(-1).to(finbert.device)
        pred_label2 = torch.tensor(list(test_res2.iloc[sample_ids.numpy()]["prediction"])).view(-1).to(finbert.device)
        logits1 = F.one_hot(pred_label1, 3)
        logits2 = F.one_hot(pred_label2, 3)
        
        # pred_label1 = torch.argmax(logits1, dim=-1).view(-1)
        # pred_label2 = torch.argmax(logits2, dim=-1).view(-1)
        real_label = torch.from_numpy(np.array(list(curr_test_df["labels"]))).view(-1).to(finbert.device)
        
        qualified_ids = torch.logical_and(pred_label2 == real_label, pred_label1 != pred_label2).nonzero().view(-1)
        
        curr_violation_count1 = apply_rules_single_per_sample_evaluation(curr_test_df, logits1, meta_class_rule).item()
        curr_violation_count2 = apply_rules_single_per_sample_evaluation(curr_test_df, logits2, meta_class_rule).item()
        if curr_violation_count1 <= curr_violation_count2:
            continue
        
        # if args.tta_method == "rule":
        for sid in range(pred_label2.shape[0]):
            curr_label1 = pred_label1[sid]
            curr_label2 = pred_label2[sid]
            violation_mapping1 = apply_rules_single_per_sample_evaluation_full(curr_test_df.iloc[sid:sid+1], logits1[sid:sid+1], meta_class_rule)
            violation_mapping2 = apply_rules_single_per_sample_evaluation_full(curr_test_df.iloc[sid:sid+1], logits2[sid:sid+1], meta_class_rule)
            for l in [0,1,2]:
                # if l == real_label[sid].item():
                #     continue
                violation_ls1 = torch.cat(list(violation_mapping1[l].values()))
                violation_ls2 = torch.cat(list(violation_mapping2[l].values()))
                if torch.sum(violation_ls1) > torch.sum(violation_ls2):
                    print()

        
        # total_violation_count += apply_rules_single_per_sample_evaluation_full(curr_test_df, logits1, meta_class_rule)

    logger.info("total rule violations::")    
    logger.info(str(total_violation_count))

def eval_test_rule_violations2(test_df, finbert, dataset_for_sampling_loader, model, testloader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, sampling_times = 1000):
    model.eval()
    with torch.no_grad():
        total_violation_count = 0
        total_violation_loss = 0
        testloader.dataset.turn_off_aug=True
        all_meta_class_pred_rule_label_mappings = dict()
        batch_size = testloader.batch_size
        test_dataset_count = len(testloader.dataset)
        for item in tqdm(testloader):
            sample_ids, batch = item
            batch = tuple(t.to(finbert.device) for t in batch)
            input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
            df = test_df.iloc[sample_ids.numpy()]

            pred = model(input_ids, attention_mask, token_type_ids)[0]
            # image, target, path, df = item
            # if type(image) is list:
            #     image = image[0]

            # if torch.cuda.is_available():
            #     image = image.cuda()
            # pred = model(image)
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

        # logging.info("total violation count::%d", total_violation_count)
        # logging.info("total_violation_loss::%f", total_violation_loss)
        
        print("total violation count::%d", total_violation_count)
        print("total_violation_loss::%f", total_violation_loss)

    model.train()
    return total_violation_count, total_violation_loss

def perturb_input_ids(input_ids, attention_mask):
    perturbed_input_ids = input_ids.clone()
    perturbed_attention_mask = attention_mask.clone()
    
    nonzero_ids = perturbed_input_ids.nonzero()
    
    rand_sample_ids = torch.randperm(len(nonzero_ids))[0:10].numpy()
    
    for idx in rand_sample_ids:
        perturbed_input_ids[nonzero_ids[idx][0],nonzero_ids[idx][1]] = 0
        perturbed_attention_mask[nonzero_ids[idx][0],nonzero_ids[idx][1]] = 0
    return perturbed_input_ids, perturbed_attention_mask

def test_time_adaptation_main(args, finbert, test_samples, test_df, bucket_features_test,  model, meta_class_rule, criterion, filtered_rules, filtered_rule_f1_scores, meta_class_mappings):
        """
        Trains the model.
        Parameters
        ----------
        examples: list
            Contains the data as a list of InputExample's
        model: BertModel
            The Bert model to be trained.
        weights: list
            Contains class weights.
        Returns
        -------
        model: BertModel
            The trained model.
        """

        # validation_examples = self.get_data('validation')

        # global_step = 0

        # self.validation_losses = []

        # Training
        test_loader_for_training, _ = get_loader(finbert, test_samples, 'train')
        
        test_loader_for_evaluation, _ = get_loader(finbert, test_samples, 'eval')
        
        init_test_violation_count = evaluate_rule_violation_count(finbert, test_loader_for_evaluation, model, test_df, meta_class_rule)
        init_test_acc = test_model(finbert, model, test_samples)
        # eval_test_rule_violations2(bucket_features_test, finbert, dataset_for_sampling_loader, model, test_loader_for_evaluation, filtered_rules, filtered_rule_f1_scores, meta_class_mappings)

        save_path = None
        if args.save_path is not None:
            save_path = os.path.join(args.save_path, args.tta_method)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

        model.train()

        step_number = len(test_loader_for_training)
        best_test_acc = -np.inf
        best_violation_count = -np.inf
        i = 0
        # for _ in trange(int(finbert.config.num_train_epochs), desc="Epoch"):
        for epoch in range(int(finbert.config.num_train_epochs)):
            
            logger.info("start training epoch::")
            logger.info(str(epoch))

            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            iterator = tqdm(enumerate(test_loader_for_training), desc="Training", total=len(test_loader_for_training))

            # for sample_ids, batch in tqdm(test_loader_for_training):
            for _, (sample_ids, batch) in iterator:

                # if (finbert.config.gradual_unfreeze and i == 0):
                #     for param in model.bert.parameters():
                #         param.requires_grad = False

                # if (step % (step_number // 3)) == 0:
                #     i += 1

                # if (finbert.config.gradual_unfreeze and i > 1 and i < finbert.config.encoder_no):

                #     for k in range(i - 1):

                #         try:
                #             for param in model.bert.encoder.layer[finbert.config.encoder_no - 1 - k].parameters():
                #                 param.requires_grad = True
                #         except:
                #             pass

                # if (finbert.config.gradual_unfreeze and i > finbert.config.encoder_no + 1):
                #     for param in model.bert.embeddings.parameters():
                #         param.requires_grad = True

                batch = tuple(t.to(finbert.device) for t in batch)

                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

                # curr_test_df = test_df.iloc[sample_ids.numpy()]

                curr_test_df = bucket_features_test.iloc[sample_ids.numpy()]

                logits = model(input_ids, attention_mask, token_type_ids)[0]
                
                if args.tta_method == "rule":
                    
                    loss = apply_rules_single_per_sample(curr_test_df, logits, meta_class_rule, criterion)
                    logits = torch.softmax(logits, dim=-1)
                    # loss = apply_rules_minibatch(curr_test_df, logits, filtered_rules, filtered_rule_f1_scores, meta_class_mappings, sigmoid_f1)
                    loss = entropy_classification_loss(logits) + loss*min(0.5*max(epoch - 23,0), 1)
                    # loss = loss
                elif args.tta_method == "memo":
                    p_input_ids1, p_attention_mask1 = perturb_input_ids(input_ids, attention_mask)
                    pred0 = model(p_input_ids1, p_attention_mask1, token_type_ids)[0]
                    p_input_ids2, p_attention_mask2 = perturb_input_ids(input_ids, attention_mask)
                    pred1 = model(p_input_ids2, p_attention_mask2, token_type_ids)[0]
                    
                    loss = memo_loss(pred0, pred1, logits)
                #     del pred0, pred1, pred2
                if args.tta_method == "tent":
                    pred = model(input_ids, attention_mask, token_type_ids)[0]
                    pred = torch.softmax(pred, dim=-1)
                    loss = entropy_classification_loss(pred)
                if args.tta_method == "cpl":
                    pred = model(input_ids, attention_mask, token_type_ids)[0]
                    loss = conjugate_pl(pred, num_classes=pred.shape[-1])
                if args.tta_method == "rpl":
                    pred = model(input_ids, attention_mask, token_type_ids)[0]
                    loss = robust_pl(pred)
                if args.tta_method == "norm":
                    pred = model(input_ids, attention_mask, token_type_ids)[0]
                    loss = 0
                    del pred
                # weights = finbert.class_weights.to(finbert.device)

                # if finbert.config.output_mode == "classification":
                #     loss_fct = CrossEntropyLoss(weight=weights)
                #     loss = loss_fct(logits.view(-1, finbert.num_labels), label_ids.view(-1))
                # elif finbert.config.output_mode == "regression":
                #     loss_fct = MSELoss()
                #     loss = loss_fct(logits.view(-1), label_ids.view(-1))

                # if finbert.config.gradient_accumulation_steps > 1:
                #     loss = loss / finbert.config.gradient_accumulation_steps
                # else:
                if type(loss) is torch.Tensor:
                    loss.backward()

                    tr_loss += loss.item()
                # else:
                    
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # if (step + 1) % finbert.config.gradient_accumulation_steps == 0:
                #     if finbert.config.fp16:
                #         lr_this_step = finbert.config.learning_rate * warmup_linear(
                #             global_step / finbert.num_train_optimization_steps, finbert.config.warm_up_proportion)
                #         for param_group in finbert.optimizer.param_groups:
                #             param_group['lr'] = lr_this_step
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                finbert.optimizer.step()
                # finbert.scheduler.step()
                finbert.optimizer.zero_grad()
                    

            # Validation

            # validation_loader, _ = self.get_loader(validation_examples, phase='eval')
            model.eval()
            
            
            test_acc = test_model(finbert, model, test_samples)            
            total_violation_count = evaluate_rule_violation_count(finbert, test_loader_for_evaluation, model, test_df, meta_class_rule)
            logger.info("test accuracy::")
            logger.info(str(test_acc))
            logger.info("test accuracy improvement::")
            ratio = (test_acc - init_test_acc)/init_test_acc
            logger.info(str(ratio))
            
            logger.info("total rule violations::")    
            logger.info(str(total_violation_count.item()))
            logger.info("reduction of total rule violation count::")
            ratio = (init_test_violation_count - total_violation_count)/init_test_violation_count
            logger.info(str(ratio.item()))
            
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_violation_count = total_violation_count
            
            # eval_test_rule_violations2(test_loader_for_evaluation, model, test_loader, filtered_rules, filtered_rule_f1_scores, meta_class_mappings)
            # eval_test_rule_violations2(bucket_features_test, finbert, dataset_for_sampling_loader, model, test_loader_for_evaluation, filtered_rules, filtered_rule_f1_scores, meta_class_mappings)
            if save_path is not None:
                torch.save(model.state_dict(), os.path.join(save_path, "model_" + str(epoch)))
            

        logger.info("best test accuracy::")
        logger.info(str(best_test_acc))
        logger.info("best test accuracy improvement::")
        ratio = (best_test_acc - init_test_acc)/init_test_acc
        logger.info(str(ratio))
        
        logger.info("best total rule violations::")    
        logger.info(str(best_violation_count.item()))
        logger.info("best reduction of total rule violation count::")
        ratio = (init_test_violation_count - best_violation_count)/init_test_violation_count
        logger.info(str(ratio.item()))


        return model

# def test_time_adaptation_main(meta_class_rule, test_loader_for_model_eval, dataset_for_sampling_loader, args, save_dir, model, optimizer, test_loader, meta_class_mappings, output_prefix="filtered_meta_class_", k = 20, validate_rule_file_name = "rule_f1_bounds_imagenetx.jsonl"):
#     # if not args.load_filtered_rules:
#     #     meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = parse_rule_file(validate_rule_file_name, k = k)
#     # else:
#     #     meta_class_pred_boolean_mappings = load_objs(os.path.join(args.cache_dir, output_prefix + "pred_boolean_mappings"))
#     #     meta_class_rule_score_mappings = load_objs(os.path.join(args.cache_dir, output_prefix + "rule_score_mappings"))
#     # meta_class_pred_boolean_mappings, meta_class_rule_score_mappings = filter_rules_by_symbolic_notations(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings)
#     #     # select_top_k_rules_per_class(meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, None, k=k)
#     # total_rule_count = sum([len(meta_class_pred_boolean_mappings[key]) for key in meta_class_pred_boolean_mappings])
#     # logging.info("total count of rules:%d", total_rule_count)
#     # criterion = torch.nn.CrossEntropyLoss()
#     # # eval_main(test_loader, model)
#     # # eval_test_rule_violations(model, test_eval_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
    
#     # eval_test_rule_violations2(dataset_for_sampling_loader, model, test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)

#     criterion = torch.nn.CrossEntropyLoss()
#     min_loss = np.inf
#     min_loss_total_rule_violation_count = 0
#     min_loss_total_rule_violation_loss = 0
    
#     for e in range(args.epochs):
#         if args.tta_method == "memo":
#             test_loader.dataset.turn_off_aug=False
        
#         total_loss = 0    
        
#         for item in tqdm(test_loader):
#             image, target, path, df = item
#             if type(image) is not list:
#                 if torch.cuda.is_available():
#                     image = image.cuda()
#             else:
#                 image = [img.cuda() for img in image]
#             if args.tta_method == "rule":
#                 pred = model(image)
#                 # pred = torch.softmax(pred, dim=-1)
#                 loss = apply_rules_single_per_sample(df, pred, meta_class_rule, criterion)
#                 # loss = apply_rules_single_per_sample(df, pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, criterion)
#                 del pred
#             elif args.tta_method == "memo":
#                 pred0 = model(image[0])
#                 pred1 = model(image[1])
#                 pred2 = model(image[2])
#                 loss = memo_loss(pred0, pred1, pred2)
#                 del pred0, pred1, pred2
#             if args.tta_method == "tent":
#                 pred = model(image)
#                 pred = torch.softmax(pred, dim=-1)
#                 loss = entropy_classification_loss(pred)
#             if args.tta_method == "cpl":
#                 pred = model(image)
#                 loss = conjugate_pl(pred, num_classes=pred.shape[-1])
#             if args.tta_method == "rpl":
#                 pred = model(image)
#                 loss = robust_pl(pred)
#             if args.tta_method == "norm":
#                 pred = model(image)
#                 loss = 0
#                 del pred
#             optimizer.zero_grad()
#             if loss > 0:
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.cpu().item()
            
            
#             del loss, image
        
        
        
#         torch.save(model.state_dict(), os.path.join(save_dir, "test_model_" + str(e)))
#         # eval_main(test_loader_for_model_eval, model)
#         # # eval_test_rule_violations(model, test_eval_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
#         # total_violation_count, total_violation_loss = eval_test_rule_violations2(dataset_for_sampling_loader, model, test_loader, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings)
#         if total_loss < min_loss:
#             min_loss = total_loss
#             min_loss_total_rule_violation_count = total_violation_count
#             min_loss_total_rule_violation_loss = total_violation_loss
#         logging.info("current min loss total violation count::%d", min_loss_total_rule_violation_count)
#         logging.info("current min loss total_violation_loss::%f", min_loss_total_rule_violation_loss)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    # 
    
    parser.add_argument('--epochs', type=int, default=50, help='used for resume')
    parser.add_argument('--lm_path', type=str, default="filtered_meta_class_", help='used for resume')
    parser.add_argument('--data_path', type=str, default="filtered_meta_class_", help='used for resume')
    parser.add_argument('--model_path', type=str, default="filtered_meta_class_", help='used for resume')
    parser.add_argument('--save_path', type=str, default=None, help='used for resume')
    parser.add_argument('--full_model',
        action='store_true',
        help='update full models')
    parser.add_argument('--lr', type=float, default=1e-3, help='used for resume')
    parser.add_argument('--tta_method', type=str, default="rule", help='used for resume')

    args = parser.parse_args()
    return args

def obtain_rules(root_dir):
    # extract bounds

    with open(os.path.join(root_dir, "bounds.csv"), "r") as bounds_file:
        reader = csv.reader(bounds_file, delimiter='\t')
        bounds = [ [ ast.literal_eval(col) for col in row ] for row in list(reader) ]
    
    with open(os.path.join(root_dir, "bounds_val.csv"), "r") as bounds_file:
        reader = csv.reader(bounds_file, delimiter='\t')
        bounds_val = [ [ ast.literal_eval(col) for col in row ] for row in list(reader) ]
        
    return bounds, bounds_val


def test_model(finbert, model, test_data):
    test_res = finbert.evaluate(model=model, examples=test_data)
    test_res['prediction'] = test_res.predictions.apply(lambda x: np.argmax(x,axis=0))
    test_res = test_res.drop(columns=['agree_levels', 'predictions'])
    test_acc = 1 - len(test_res.loc[test_res["labels"] != test_res["prediction"]]) / len(test_res)
    return test_acc


def remove_common_part(class_rule_mappings):
    
    new_class_rule_mappings = {0:[], 1:[], 2:[]}
    
    for idx in range(len(class_rule_mappings[0])):
        rule_from_class0 = class_rule_mappings[0][idx]
        rule_from_class1 = class_rule_mappings[1][idx]
        rule_from_class2 = class_rule_mappings[2][idx]
        
        assert set(rule_from_class0.keys()) == set(rule_from_class1.keys())
        assert set(rule_from_class2.keys()) == set(rule_from_class1.keys())
        
        new_rule_class0 = dict()
        new_rule_class1 = dict()
        new_rule_class2 = dict()
        for key in rule_from_class0:
            bound0 = rule_from_class0[key]
            bound1 = rule_from_class1[key]
            bound2 = rule_from_class2[key]
            r0 = IntervalSet([Interval(bound0[0], bound0[1])])
            r1 = IntervalSet([Interval(bound1[0], bound1[1])])
            r2 = IntervalSet([Interval(bound2[0], bound2[1])])
            remaining_r0 = r0 - r1 - r2
            # remaining_r0 = remaining_r0 - r2
            remaining_r1 = r1 - r0 - r2
            # remaining_r1 = remaining_r1 - r2
            remaining_r2 = r2 - r1 - r0
            # remaining_r2 = remaining_r2 - r1
            
            if len(remaining_r0) > 0:
                new_rule_class0[key] = [remaining_r0[0].lower_bound, remaining_r0[0].upper_bound]
            else:
                new_rule_class0[key] = []
            if len(remaining_r1) > 0:
                new_rule_class1[key] = [remaining_r1[0].lower_bound, remaining_r1[0].upper_bound]
            else:
                new_rule_class1[key] = []
            
            if len(remaining_r2) > 0:
                new_rule_class2[key] = [remaining_r2[0].lower_bound, remaining_r2[0].upper_bound]
            else:
                new_rule_class2[key] = []
            
        new_class_rule_mappings[0].append(new_rule_class0)
        new_class_rule_mappings[1].append(new_rule_class1)
        new_class_rule_mappings[2].append(new_rule_class2)
            
    return new_class_rule_mappings
def pre_process_rules(bounds):
    class_rule_mappings = dict()
    
    for bound in bounds:
        label = bound[0][1]
        
        bound_ls = dict()
        
        bound_ls[bound[1][0]]= [bound[1][1], bound[1][2]]
        
        if len(bound) == 3:
            bound_ls[bound[2][0]]= [bound[2][1], bound[2][2]]
            # continue
            
        if not label in class_rule_mappings:
            class_rule_mappings[label] = []
        
        class_rule_mappings[label].append(bound_ls)
        
    remaining_class_rule_mappings = class_rule_mappings
    remaining_class_rule_mappings = remove_common_part(class_rule_mappings)
    return remaining_class_rule_mappings

def validate_rules(label_rule_mappings, label_rule_mappings_val):
    
    filtered_label_rule_mappings = dict()
    
    for label in label_rule_mappings:
        filtered_label_rule_mappings[label] = []
        rule_ls = label_rule_mappings[label]
        rule_val_ls = label_rule_mappings_val[label]
        assert len(rule_ls) == len(rule_val_ls)
        
        for idx in range(len(rule_ls)):
            rule = rule_ls[idx]
            rule_val = rule_val_ls[idx]
            overlap = 1
            for key in rule:
                assert key in rule_val
                bound1 = rule[key]
                bound2 = rule_val[key]
                if len(bound1) <= 0 or len(bound2) <= 0:
                    overlap = 0
                    break
                
                min_lb = min(bound2[0], bound1[0])
                max_lb = max(bound2[0], bound1[0])
                min_hb = min(bound2[1], bound1[1])
                max_hb = max(bound2[1], bound1[1])
                if min_hb < max_lb:
                    overlap *= 0
                elif max_hb == min_lb:
                    overlap *= 1
                else:
                    overlap *= (min_hb - max_lb)/(max_hb - min_lb)
                    
            # if overlap > 0.8:
            if overlap > 0.2:
                filtered_label_rule_mappings[label].append(rule)
            
    return filtered_label_rule_mappings


def main(args):
    
    # root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    
    lm_path = os.path.join(args.model_path, 'language_model/finbertTRC2')
    cl_path = os.path.join(args.model_path, 'classifier_model/finbert-sentiment')
    data_path = args.data_path# root_dir + '/data/sentiment_data'

    bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)
    config = Config(data_dir=data_path,
                    bert_model=bertmodel,
                    num_train_epochs=args.epochs,
                    model_dir=lm_path,
                    max_seq_length = 48,
                    train_batch_size = 32,
                    learning_rate = args.lr,
                    output_mode='classification',
                    warm_up_proportion=0.2,
                    local_rank=-1,
                    discriminate=True,
                    gradual_unfreeze=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
    finbert = FinBert(config)
    finbert.base_model = 'bert-base-uncased'
    finbert.config.discriminate=True
    finbert.config.gradual_unfreeze=True
    finbert.prepare_model(label_list=['positive','negative','neutral'])
    
    test_data = finbert.get_data('test')
    # if args.cached_model_path is not None:
    #     model.load_state_dict(torch.load(args.cached_model_path))



    model = model.to(finbert.device)
    # train_data = finbert.get_data('train')
    # val_data = finbert.get_data('validation')
    # test_res = test_model(finbert, model, test_data)

    bounds, bounds_val = obtain_rules(args.data_path)
    
    label_rule_mappings = pre_process_rules(bounds)
    
    label_rule_mappings_val = pre_process_rules(bounds_val)
    
    filtered_label_rule_mappings = validate_rules(label_rule_mappings, label_rule_mappings_val)
    
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
    
    finbert.optimizer = optimizer
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # features_test = pd.read_csv(os.path.join(root_dir,'notebooks/finbert_test_features.csv'), sep='\t', index_col=0)
    features_test = pd.read_csv(os.path.join(args.data_path,'finbert_test_features.csv'), sep='\t', index_col=0)
    
    # with open(os.path.join(root_dir,'bucket_test_features'), "rb") as f:
    #     bucket_features_test = pickle.load(f)
    
    # with open(os.path.join(root_dir, "filtered_rules"), "rb") as f:
    #     filtered_rules = pickle.load(f)
    # with open(os.path.join(root_dir, "filtered_rule_f1_scores"), "rb") as f:
    #     filtered_rule_f1_scores = pickle.load(f)
        
    # meta_class_mappings = {0:0,1:1,2:2}
    
    # test_time_adaptation_main(args, finbert, test_data, features_test, bucket_features_test,  model, None, criterion, filtered_rules, filtered_rule_f1_scores, meta_class_mappings)
    
    # test time adaptation
    test_time_adaptation_main(args, finbert, test_data, features_test, features_test,  model, filtered_label_rule_mappings, criterion, None, None, None)
    
    
    '''compare two models for qualitative evaluations'''
    # # trained_model = finbert.train(train_examples = test_data, model = model)
    # test_loader_for_evaluation, _ = get_loader(finbert, test_data, 'eval')
    # # compare two models
    # model1 = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
    
    # model1.load_state_dict(torch.load("/data6/wuyinjun/finbert/rule/model_0"))
    # model1 = model1.to(finbert.device)
    # model2 = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
    # model2.load_state_dict(torch.load("/data6/wuyinjun/finbert/rule/model_5"))
    # model2 = model2.to(finbert.device)
    # model1 = tent.configure_model(model1)
    # model2 = tent.configure_model(model2)
    # test_res1 = test_model(finbert, model1, test_data)
    # # evaluate_rule_violation_count(finbert, test_loader_for_evaluation, model, features_test, filtered_label_rule_mappings)

    # test_res2 = test_model(finbert, model2, test_data)
    # # evaluate_rule_violation_count(finbert, test_loader_for_evaluation, model2, features_test, filtered_label_rule_mappings)
    
    # evaluate_rule_violation_count_compare(finbert, test_loader_for_evaluation, test_res1, test_res2, features_test, filtered_label_rule_mappings)
    return
    

if __name__ == '__main__':
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    main(args)