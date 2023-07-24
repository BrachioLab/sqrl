import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from rule_processing.sigmoidF1 import sigmoidF1
from rule_processing.process_rules import apply_rules_minibatch
from baseline_methods import memo_loss, entropy_classification_loss, conjugate_pl, robust_pl
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
from torch.utils.data import RandomSampler, DataLoader
import numpy as np
from scipy.stats import bootstrap

# bp_f1_bounds = [[0.9012745,0.944088], [0.801068, 0.822715]]
# bp_f1_bounds = [(0.8931240531896988, 0.9528272351867858), (0.7974638951878686, 0.827757063228472), (0.8286166033199962, 0.8676921144765409)]

# bp_f1_bounds = [(0.8906351183063512, 0.9533333997837041), (0.7935793747933477, 0.8281283674007633), (0.8297505126452495, 0.8686815589031225)]
bp_f1_bounds = [(0.8851551993852403, 0.9560966496450367), (0.7917903776657402, 0.8308253609708567), (0.8273053803816193, 0.87175089695251)]

def apply_bp_rule(feat, normalizer, rule_class = 1):
    orig_feat = normalizer.inverse_transform(feat.cpu().numpy())
    orig_feat = torch.from_numpy(orig_feat)
    if torch.cuda.is_available():
        orig_feat = orig_feat.cuda()
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = (orig_feat[:,4] >= 160)
    

    return satisfied_sample_ids, rule_class


def apply_bp_rule_medium(feat, normalizer, rule_class = 1):
    orig_feat = normalizer.inverse_transform(feat.cpu().numpy())
    orig_feat = torch.from_numpy(orig_feat)
    if torch.cuda.is_available():
        orig_feat = orig_feat.cuda()
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = torch.logical_and((orig_feat[:,4] < 160), (orig_feat[:,4] > 120))
    

    return satisfied_sample_ids, rule_class


def apply_bp_rule_negative(feat, normalizer, rule_class = 0):
    orig_feat = normalizer.inverse_transform(feat.cpu().numpy())
    orig_feat = torch.from_numpy(orig_feat)
    if torch.cuda.is_available():
        orig_feat = orig_feat.cuda()
    # satisfied_sample_ids = torch.logical_and((feat[:,2]*feat_std[2] + feat_mean[2] > 110), (feat[:,4]*feat_std[4] + feat_mean[4] > 160))
    satisfied_sample_ids = (orig_feat[:,4] <= 120)
    

    return satisfied_sample_ids, rule_class

def differentiable_blood_pressure_rule(rule_func, feat_num, feat_cat, pred, normalizer, sigmoidF1, F1_score_low, F1_score_high):
    satisfied_sample_ids, expected_label = rule_func(feat_num, normalizer)
    if torch.sum(satisfied_sample_ids) <= 0:
        return 0
    satisfied_feat = feat_num[satisfied_sample_ids]
    satisfied_pred = pred[satisfied_sample_ids].view(-1)

    # satisfied_pred = (satisfied_pred > 0.5).long().view(-1)
    expected_pred = torch.ones_like(satisfied_pred)
    if expected_label == 1:
        F1_score = sigmoidF1(satisfied_pred, expected_pred)
    else:
        F1_score = sigmoidF1(1 - satisfied_pred, expected_pred)


    bounds = torch.tensor([F1_score_low, F1_score_high])

    if F1_score < bounds[0]:
        loss = torch.abs(F1_score - bounds[0])
    elif F1_score > bounds[1]:
        loss = torch.abs(F1_score - bounds[1])
    else:
        loss = torch.tensor(0)
    loss = torch.clamp(loss, min=0.0, max=1.0)
    # print("Aspect Ratio Loss:", loss, aspect_ratio, bounds)
    # loss = torch.tanh(loss)
    
    return loss


def get_rule_ls():
    rule_ls = [apply_bp_rule, apply_bp_rule_negative, apply_bp_rule_medium]
    return rule_ls

def compute_rule_loss(feat_num, feat_cat, pred, normalizer, sigmoidF1):
    # loss = weight_blood_pressure_rule(feat, pred, mean_feat, std_feat)
    rule_ls = get_rule_ls()
    loss = 0
    for rule_idx in range(len(rule_ls)):
        rule_func = rule_ls[rule_idx]
        loss += differentiable_blood_pressure_rule(rule_func, feat_num, feat_cat, pred, normalizer, sigmoidF1, bp_f1_bounds[rule_idx][0], bp_f1_bounds[rule_idx][1])
    return loss

def evaluate_rule_violations_backup(feat, pred, normalizer):
    rule_ls = get_rule_ls()
    violation_count = 0
    loss = 0
    for rule_idx in range(len(rule_ls)):
        rule_func = rule_ls[rule_idx]
        satisfied_sample_ids, expected_label = rule_func(feat, normalizer)
        satisfied_pred = pred[satisfied_sample_ids].view(-1)
        expected_pred = torch.ones_like(satisfied_pred)
        if len(satisfied_pred) <= 0:
            continue

        if expected_label == 1:
            f1_score_val = f1_score(expected_pred.cpu().numpy(), satisfied_pred.cpu().numpy())
        else:
            f1_score_val = f1_score(expected_pred.cpu().numpy(), (1-satisfied_pred).cpu().numpy())
        if f1_score_val >= bp_f1_bounds[rule_idx][0] and f1_score_val <= bp_f1_bounds[rule_idx][1]:
            violation_count += 0
        else:
            violation_count += 1
            if f1_score_val < bp_f1_bounds[rule_idx][0]:
                loss += (bp_f1_bounds[rule_idx][0] - f1_score_val)
            elif f1_score_val > bp_f1_bounds[rule_idx][1]:
                loss += (f1_score_val - bp_f1_bounds[rule_idx][1])
    return violation_count, loss

def default_test_time_adaptation_loop(cfg, net, trainloader, train_setup, device, normalizer, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings):
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    # criterion = train_setup.criterions
    criterion = sigmoidF1()
    train_loss = 0
    total = 0

    for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(trainloader, leave=False)):
        if type(inputs_num) is list:
            inputs_num = [inputs_num[idx].to(device).float() for idx in range(len(inputs_num))]
            inputs_num = [inputs_num[idx] if inputs_num[idx].nelement() != 0 else None for idx in range(len(inputs_num))]
            inputs_cat, targets = inputs_cat.to(device), targets.to(device)
            inputs_cat = inputs_cat if inputs_cat.nelement() != 0 else None
        else:
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                    inputs_cat if inputs_cat.nelement() != 0 else None
        optimizer.zero_grad()
        

        # entropy_loss = -torch.mean(torch.sum(torch.log(full_pred)*full_pred, dim=1))

        # loss = compute_rule_loss(inputs_num, inputs_cat, outputs, normalizer, criterion)
        if cfg.hyp.tta_method=='rule':
            outputs = torch.sigmoid(net(inputs_num, inputs_cat))

            full_pred = torch.stack([1-outputs, outputs], dim=1)
            loss = apply_rules_minibatch(df, full_pred, meta_class_pred_boolean_mappings, meta_class_rule_score_mappings, meta_class_mappings, criterion)
        elif cfg.hyp.tta_method=='memo':
            output0 = net(inputs_num[0], inputs_cat)
            output1 = net(inputs_num[1], inputs_cat)
            output2 = net(inputs_num[2], inputs_cat)

            loss = memo_loss(output0, output1, output2)
        elif cfg.hyp.tta_method=='tent':
            outputs = torch.sigmoid(net(inputs_num, inputs_cat))

            full_pred = torch.stack([1-outputs, outputs], dim=1)
            loss = entropy_classification_loss(full_pred)
            
        if cfg.hyp.tta_method == "cpl":
            outputs = torch.sigmoid(net(inputs_num, inputs_cat))
            full_pred = torch.stack([1-outputs, outputs], dim=1)
            loss = conjugate_pl(full_pred, num_classes=full_pred.shape[-1], input_probs=True)
        if cfg.hyp.tta_method == "rpl":
            outputs = torch.sigmoid(net(inputs_num, inputs_cat))
            full_pred = torch.stack([1-outputs, outputs], dim=1)
            loss = robust_pl(full_pred, input_probs=True)
            
        if cfg.hyp.tta_method == "norm":
            pred = net(inputs_num, inputs_cat)
            loss = 0
        # loss = entropy_loss

        if type(loss) is torch.Tensor:
            # loss = criterion(outputs.view(targets.shape), targets)
            loss.backward()
            optimizer.step()
            # print("loss::", loss.cpu().item())
            train_loss += loss.cpu().item()
        
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss


def learning_statistics_for_bp_rule(batch_size, dataset, normalizer):
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset)*200)
    # dl = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    dl = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    f1_ls = []

    precision_ls = []

    recall_ls = []

    # for (idx, feat, labels) in tqdm(dl):
    #     if torch.cuda.is_available():
    #         feat = feat.cuda()
    #         labels = labels.cuda()

    for batch_idx, (inputs_num, inputs_cat, targets, df) in enumerate(tqdm(dl, leave=False)):
        # inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                 inputs_cat if inputs_cat.nelement() != 0 else None

        satisfied_sample_ids, _ = apply_bp_rule(inputs_num, normalizer)
        # pred = model(feat)
        # satisfied_pred = pred[satisfied_sample_ids]

        # pred_labels = (satisfied_pred > 0.5).long().reshape(-1)
        selected_targets = targets[satisfied_sample_ids].view(-1)
        expected_label_tensor = torch.ones_like(selected_targets)
        # y_true = expected_label_tensor.view(-1) == pred_labels.view(-1)

        f1 = f1_score(expected_label_tensor.cpu().numpy(), selected_targets.cpu().numpy())

        # precision, recall,_,_ = precision_recall_fscore_support(expected_label_tensor.cpu().numpy(), pred_labels.cpu().numpy())

        f1_ls.append(f1)

            # precision_ls.append(precision)

            # recall_ls.append(recall)

    f1_CI = calculate_confidence_interval(f1_ls)
    # f1_mean = np.mean(np.array(f1_ls))
    return f1_CI.low, f1_CI.high

def calculate_confidence_interval(stat_ls):
    stat_array = np.array(stat_ls)
    rng = np.random.default_rng()
    res = bootstrap((stat_array,), np.mean, confidence_level=0.99,
                random_state=rng)

    return res.confidence_interval