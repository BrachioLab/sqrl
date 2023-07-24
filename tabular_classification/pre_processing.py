import os, sys
from re import L
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import numpy as np

import argparse
from scipy.stats import bootstrap

import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from rule_processing.sigmoidF1 import sigmoidF1
from tqdm import tqdm

continue_feats=["age", "height", "weight", "ap_lo", "ap_hi"]
discrete_feats=["cholesterol", "gluc"]
bin_discrete_feats=["smoke", "alco", "active"]
target_col = ["cardio"]

bp_f1_bounds = [0.9265652493348585,0.928723631181347]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--epochs', type=int, default=200, help='used for resume')
    # parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    # parser.add_argument('--lr', type=float, default=0.02, help='used for resume')

    parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    parser.add_argument('--lr', type=float, default=0.002, help='used for resume')
    parser.add_argument('--model', type=str, default=0.02, help='used for resume', choices=["mlp", "dd"])
    parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--work_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--data_dir', type=str, default="out/", help='used for resume')

    args = parser.parse_args()
    return args


class MLP(nn.Module):
    def __init__(self, input_feat, out_feat=1):
        super(MLP,self).__init__()
        self.out_feat = out_feat
        self.fc1 = nn.Linear(input_feat, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, out_feat)

    def forward(self, X):
        out = F.relu(self.fc1(X))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if self.out_feat == 1:
            out = torch.sigmoid(out)
        return out


class Card_dataset(Dataset):
    def __init__(self, data, labels):
        self.feat = data
        self.labels = labels
    
    def __getitem__(self, index):
        return index, self.feat[index], self.labels[index]
    
    def __len__(self):
        return len(self.feat)

    @staticmethod
    def collate_fn(data):
        index_ls = [data[i][0] for i in range(len(data))] 
        feat_ls = [data[i][1] for i in range(len(data))]
        label_ls = [data[i][2] for i in range(len(data))]

        feat_tensor = torch.stack(feat_ls)
        label_tensor = torch.tensor(label_ls)
        index_tensor = torch.tensor(index_ls)
        return index_tensor, feat_tensor, label_tensor


def read_tab_data(file_name):
    data = pd.read_csv(file_name, delimiter=";")
    return data

def filter_unexpected_blood_pressure(data):
    data = data[(data["ap_hi"] < 200) & (data["ap_hi"] > 0)]
    data = data[(data["ap_lo"] < 200) & (data["ap_lo"] > 0)]
    data = data[data["ap_hi"] > data["ap_lo"]]
    return data

def split_by_gender(data):
    male_data = data[data["gender"] == 1]
    female_data = data[data["gender"] == 2]
    return male_data, female_data

def transform_onehot(x):
    if x == 1:
        return [0,0,1]
    elif x == 2:
        return [0,1,0]
    elif x == 3:
        return [1,0,0]

def construct_tensor_from_df(df):
    tensor1 = torch.tensor(df[continue_feats].values.tolist())

    tensor2_ls = []

    for attr in discrete_feats:
        curr_ls = list(df[attr])
        transformed_ls = [transform_onehot(x) for x in curr_ls]
        tensor2_ls.append(torch.tensor(transformed_ls))

    tensor2 = torch.cat(tensor2_ls, dim=1)    

    tensor3 = torch.tensor(df[bin_discrete_feats].values.tolist())

    all_tensor = torch.cat([tensor1, tensor2, tensor3], dim=1)

    target = torch.tensor(df[target_col].values.tolist())

    return all_tensor, target


def evaluate_main(test_loader, model):
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_sample_count = 0
        pred_ls = []
        label_ls = []
        for (idx, feat, labels) in test_loader:
            if torch.cuda.is_available():
                feat = feat.cuda()
                labels = labels.cuda()
            pred = model(feat)
            pred_labels = (pred > 0.5).long()
            label_ls.append(labels.view(-1))
            pred_ls.append(pred_labels.view(-1))
            correct = torch.sum(pred_labels.view(-1) == labels.view(-1))
            total_correct += correct
            total_sample_count += len(labels)

        print("Accuracy::", total_correct*1.0/total_sample_count)
        pred_ls_tensor = torch.cat(pred_ls)
        label_ls_tensor = torch.cat(label_ls)
        print(classification_report(pred_ls_tensor.cpu().numpy(), label_ls_tensor.cpu().numpy()))


    model.train()
    return label_ls_tensor.cpu().numpy()

def train_main(train_loader, test_loader, model, epochs, criterion, optimizer):
    for e in range(epochs):
        for (idx, feat, labels) in train_loader:
            if torch.cuda.is_available():
                feat = feat.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            pred = model(feat)

            loss = criterion(pred.view(-1), labels.float().view(-1))

            loss.backward()

            optimizer.step()

        evaluate_main(test_loader, model)
        save_model(model, os.path.join(args.work_dir, "model_" + str(e)))
    test_labels = evaluate_main(test_loader, model)
    return test_labels


def normalize_data(feat_tensor, mean_feat=None, std_feat=None):
    if mean_feat is None or std_feat is None:
        mean_feat = torch.mean(feat_tensor, dim=0)
        std_feat = torch.std(feat_tensor, dim=0)
    
    feat_tensor = (feat_tensor - mean_feat.view(1,-1))/std_feat.view(1,-1)
    return feat_tensor, mean_feat, std_feat

def normalize_card(male_tensor, female_tensor):
    sub_male_tensor = male_tensor[:, 0:len(continue_feats)]
    sub_female_tensor = female_tensor[:, 0:len(continue_feats)]
    sub_male_tensor, mean_feat, std_feat =  normalize_data(sub_male_tensor)

    male_tensor[:, 0:len(continue_feats)] = sub_male_tensor

    sub_female_tensor, _, _ =  normalize_data(sub_female_tensor, mean_feat, std_feat)
    female_tensor[:, 0:len(continue_feats)] = sub_female_tensor
    return male_tensor, female_tensor, mean_feat, std_feat


def apply_bp_rule(feat, feat_mean, feat_std):
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = (feat[:,4]*feat_std[4] + feat_mean[4] >= 160)

    return satisfied_sample_ids, 1#torch.ones_like(satisfied_sample_ids)


def apply_bp_rule_medium(feat, feat_mean, feat_std):
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = torch.logical_and((feat[:,4]*feat_std[4] + feat_mean[4] < 160), (feat[:,4]*feat_std[4] + feat_mean[4] > 120))

    return satisfied_sample_ids, 1#torch.ones_like(satisfied_sample_ids)


def apply_bp_rule_negative(feat, feat_mean, feat_std):
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = (feat[:,4]*feat_std[4] + feat_mean[4] <= 120)

    return satisfied_sample_ids, 0#torch.zeros_like(satisfied_sample_ids)


def weight_blood_pressure_rule(feat, pred, feat_mean, feat_std):
    satisfied_sample_ids, _ = apply_bp_rule(feat, feat_mean, feat_std)
    if torch.sum(satisfied_sample_ids) <= 0:
        return 0
    satisfied_feat = feat[satisfied_sample_ids]
    satisfied_pred = pred[satisfied_sample_ids]
    loss = 0
    # if len(satisfied_pred) > 0:
    negative_samples = torch.nonzero(satisfied_pred < 0.5)
    if len(negative_samples) <= 0:
        return 0
    loss = torch.nn.BCELoss(satisfied_pred[negative_samples].view(-1), torch.ones_like(satisfied_pred[negative_samples]).view(-1))
    return loss


def differentiable_blood_pressure_rule(feat, pred, feat_mean, feat_std, sigmoidF1, F1_score_low, F1_score_high):
    satisfied_sample_ids, _ = apply_bp_rule(feat, feat_mean, feat_std)
    if torch.sum(satisfied_sample_ids) <= 0:
        return 0
    satisfied_feat = feat[satisfied_sample_ids]
    satisfied_pred = pred[satisfied_sample_ids].view(-1)

    # satisfied_pred = (satisfied_pred > 0.5).long().view(-1)
    expected_pred = torch.ones_like(satisfied_pred)
    F1_score = sigmoidF1(satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])

    loss = (F1_score - bounds[0]) * (F1_score - bounds[1])
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return loss


def compute_rule_loss(feat, pred, mean_feat, std_feat, sigmoidF1):
    # loss = weight_blood_pressure_rule(feat, pred, mean_feat, std_feat)    
    loss = differentiable_blood_pressure_rule(feat, pred, mean_feat, std_feat, sigmoidF1, bp_f1_bounds[0], bp_f1_bounds[1])
    return loss


def test_adaptation_main(test_loader, model, epochs, optimizer, mean_feat, std_feat, lamb = 1):

    loss_fc = sigmoidF1()

    for e in range(epochs):
        violation_times = 0

        for (idx, feat, _) in test_loader:
            if torch.cuda.is_available():
                feat = feat.cuda()
            pred = model(feat)

            full_pred = torch.cat([1-pred, pred], dim=1)

            entropy_loss = -torch.mean(torch.sum(torch.log(full_pred)*full_pred, dim=1))

            loss = compute_rule_loss(feat, pred, mean_feat, std_feat, loss_fc)

            full_loss = entropy_loss + loss*0.5
            # if type(loss) is torch.Tensor:
            print("loss::", loss.cpu().item())
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()
            # violation_times += 1


        # print("violation count::", violation_times)
        evaluate_main(test_loader, model)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path,map_location=torch.device("cpu")))

def write_pred_results(df, pred_labels, output_path):
    df["pred"] = pred_labels
    df.to_csv(output_path, sep="\t", header=False)

def np_percentile_lower(array, axis=-1):
    return np.percentile(array, q=1, axis=axis)

def np_percentile_upper(array, axis=-1):
    return np.percentile(array, q=99, axis=axis)




def calculate_confidence_interval2(stat_ls, quantile=0.5):
    stat_array = np.array(stat_ls)
    sorted_stat_array = np.sort(stat_array)
    lower_bound = np.percentile(sorted_stat_array, quantile)
    upper_bound = np.percentile(sorted_stat_array, 100-quantile)
    return lower_bound, upper_bound


def learning_statistics_for_bp_rule(args, model, male_dataset, feat_mean, feat_std):
    sampler = RandomSampler(male_dataset, replacement=True, num_samples=len(male_dataset)*100)
    dl = DataLoader(male_dataset, sampler=sampler, batch_size=args.batch_size)

    f1_ls = []

    precision_ls = []

    recall_ls = []

    rule_func_ls = [apply_bp_rule, apply_bp_rule_negative, apply_bp_rule_medium]

    f1_ls = [[] for k in range(len(rule_func_ls))]

    for (idx, feat, labels) in tqdm(dl):
        if torch.cuda.is_available():
            feat = feat.cuda()
            labels = labels.cuda()



            for rule_idx in range(len(rule_func_ls)):
                rule_func = rule_func_ls[rule_idx]

                satisfied_sample_ids, expected_label = rule_func(feat, feat_mean, feat_std)
                pred = model(feat)
                satisfied_pred = pred[satisfied_sample_ids]

                pred_labels = (satisfied_pred > 0.5).long().reshape(-1)

                expected_label_tensor = torch.ones_like(pred_labels)
                # y_true = expected_label_tensor.view(-1) == pred_labels.view(-1)

                if len(expected_label_tensor) <= 0:
                    continue

                if expected_label == 1:
                    f1 = f1_score(expected_label_tensor.cpu().numpy(), labels[satisfied_sample_ids].view(-1).cpu().numpy())
                else:
                    f1 = f1_score(expected_label_tensor.cpu().numpy(), (1 - labels[satisfied_sample_ids]).view(-1).cpu().numpy())

                # precision, recall,_,_ = precision_recall_fscore_support(expected_label_tensor.cpu().numpy(), pred_labels.cpu().numpy())

                f1_ls[rule_idx].append(f1)

            # precision_ls.append(precision)

            # recall_ls.append(recall)

    # f1_CI = calculate_confidence_interval(f1_ls)
    # # f1_mean = np.mean(np.array(f1_ls))
    # return f1_CI.low, f1_CI.high

    f1_bound_ls = []
    for sub_f1_ls in f1_ls:
        f1_low_low, f1_low_high, f1_high_low, f1_high_high = calculate_confidence_interval(sub_f1_ls)
        f1_bound_ls.append(((f1_low_low + f1_low_high)/2, (f1_high_high + f1_high_low)/2))
        print("f1 low::", (f1_low_low + f1_low_high)/2)
        print("f1 high::", (f1_high_high + f1_high_low)/2)

    print(f1_bound_ls)
    return f1_bound_ls
    # precision_CI = calculate_confidence_interval(precision_ls)
    # recall_CI = calculate_confidence_interval(recall_ls)

    
    # f1_low = f1_CI.low + 


def split_cat_numeric_data(dir, data):
    # data = data.set_index("id")
    cat_data = data[discrete_feats + bin_discrete_feats]
    num_data = data[continue_feats]
    target_data = data[target_col]

    cat_data_file_name = os.path.join(dir, "C.csv")
    num_data_file_name = os.path.join(dir, "N.csv")
    target_data_file_name = os.path.join(dir, "y.csv")

    cat_data.to_csv(cat_data_file_name,index=False)
    num_data.to_csv(num_data_file_name,index=False)
    target_data.to_csv(target_data_file_name,index=False)

def main(args):
    file_name = os.path.join(args.data_dir, "cardio_train.csv")
    data = read_tab_data(file_name)
    data = filter_unexpected_blood_pressure(data)
    split_cat_numeric_data(args.data_dir, data)
    male_data, female_data = split_by_gender(data)
    orig_male_tensor, male_target = construct_tensor_from_df(male_data)
    orig_female_tensor, female_target = construct_tensor_from_df(female_data)

    # male_tensor, female_tensor, mean_feat, std_feat = normalize_card(orig_male_tensor.clone(), orig_female_tensor.clone())
# 
    # male_dataset = Card_dataset(male_tensor, male_target)
    # female_dataset = Card_dataset(female_tensor, female_target)

    # male_dataloader = DataLoader(
    #     male_dataset,
    #     batch_size=args.batch_size,
    #     # num_workers=16, #args.num_workers,
    #     pin_memory=True,
    #     shuffle=True,
    #     collate_fn=Card_dataset.collate_fn
    # )

    # model = MLP(len(continue_feats) + 3*len(discrete_feats) + len(bin_discrete_feats))
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # female_dataloader = DataLoader(
    #     female_dataset,
    #     batch_size=args.batch_size,
    #     # num_workers=16, #args.num_workers,
    #     pin_memory=True,
    #     shuffle=False,
    #     collate_fn=Card_dataset.collate_fn
    # )

    # criterion = torch.nn.BCELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # if not os.path.exists(args.work_dir):
    #     os.makedirs(args.work_dir)

    # if args.train:
    #     female_labels = train_main(male_dataloader, female_dataloader, model, args.epochs, criterion, optimizer)
    #     save_model(model, os.path.join(args.work_dir, "last_model"))
    #     test_out_path = os.path.join(args.work_dir, "test_db")
    #     if not os.path.exists(test_out_path):
    #         os.makedirs(test_out_path)
    #     write_pred_results(female_data, female_labels, os.path.join(test_out_path, "feat_with_pred.csv"))
    # else:
    #     load_model(model, os.path.join(args.work_dir, "last_model"))
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    #     f1_bound_ls = learning_statistics_for_bp_rule(args, model, male_dataset, mean_feat, std_feat)
    #     # print("f1_low::", f1_low)
    #     # print("f1_high::", f1_high)
    #     # bp_f1_bounds[0] = f1_low
    #     # bp_f1_bounds[1] = f1_high
    #     # evaluate_main(female_dataloader, model)
    #     # female_dataloader = DataLoader(
    #     #     female_dataset,
    #     #     batch_size=args.batch_size,
    #     #     # num_workers=16, #args.num_workers,
    #     #     pin_memory=True,
    #     #     shuffle=True,
    #     #     collate_fn=Card_dataset.collate_fn
    #     # )
    #     # evaluate_main(female_dataloader, model)
    #     # test_adaptation_main(female_dataloader, model, args.epochs, optimizer, mean_feat, std_feat)
    #     # test_out_path = os.path.join(args.work_dir, "test_adapt_db")
    #     # if not os.path.exists(test_out_path):
    #     #     os.makedirs(test_out_path)
    #     # write_pred_results(female_data, female_labels, os.path.join(test_out_path, "feat_with_pred.csv"))


    # print()


if __name__ == '__main__':
    args = parse_args()
    main(args)
