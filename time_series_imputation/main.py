import argparse
import torch

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gain import *
from models.mtan.models import enc_mtan_rnn, dec_mtan_rnn, create_classifier, mtan_model_full
from models.CSDI.main_models import CSDI_Physio
from models.SAITS.SA_models import SAITS
from sklearn import metrics

from tqdm import tqdm
from util import *
from parse_args import *
from mTan_dataset import *
# import time
import Tent.Tent as tent
from argparse import Namespace
from physionet_dataset import get_dataloader, Physio_Dataset

from baseline_methods.baseline import *
import yaml

attr_ls = ['Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C

def train(train_X, model, opt, Dim, p_hint, mb_size = 128):
    
    trainM = (train_X == train_X)

    for it in tqdm(range(5000)):    
        
        #%% Inputs
        mb_idx = sample_idx(train_X.shape[0], mb_size)
        X_mb = train_X[mb_idx,:]  
        
        Z_mb = sample_Z(mb_size, Dim) 
        M_mb = trainM[mb_idx,:]  
        H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
        H_mb = M_mb * H_mb1
        
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
        X_mb = X_mb.cuda()
        M_mb = M_mb.cuda()
        H_mb = H_mb.cuda()
        New_X_mb = New_X_mb.cuda()
        # if use_gpu is True:
        #     X_mb = torch.tensor(X_mb, device="cuda")
        #     M_mb = torch.tensor(M_mb, device="cuda")
        #     H_mb = torch.tensor(H_mb, device="cuda")
        #     New_X_mb = torch.tensor(New_X_mb, device="cuda")
        # else:
        #     X_mb = torch.tensor(X_mb)
        #     M_mb = torch.tensor(M_mb)
        #     H_mb = torch.tensor(H_mb)
        #     New_X_mb = torch.tensor(New_X_mb)
        
        opt.zero_grad()
        D_loss_curr = model.discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
        D_loss_curr.backward()
        opt.step()
        
        opt.zero_grad()
        G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = model.generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
        G_loss_curr.backward()
        opt.step()    
            
        #%% Intermediate Losses
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
            print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
            print()


def train(train_X, model, opt, Dim, p_hint, mb_size = 128):
    
    trainM = (train_X == train_X)

    for it in tqdm(range(5000)):    
        
        #%% Inputs
        mb_idx = sample_idx(train_X.shape[0], mb_size)
        X_mb = train_X[mb_idx,:]  
        
        Z_mb = sample_Z(mb_size, Dim) 
        M_mb = trainM[mb_idx,:]  
        H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
        H_mb = M_mb * H_mb1
        
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
        X_mb = X_mb.cuda()
        M_mb = M_mb.cuda()
        H_mb = H_mb.cuda()
        New_X_mb = New_X_mb.cuda()
        # if use_gpu is True:
        #     X_mb = torch.tensor(X_mb, device="cuda")
        #     M_mb = torch.tensor(M_mb, device="cuda")
        #     H_mb = torch.tensor(H_mb, device="cuda")
        #     New_X_mb = torch.tensor(New_X_mb, device="cuda")
        # else:
        #     X_mb = torch.tensor(X_mb)
        #     M_mb = torch.tensor(M_mb)
        #     H_mb = torch.tensor(H_mb)
        #     New_X_mb = torch.tensor(New_X_mb)
        
        opt.zero_grad()
        D_loss_curr = model.discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
        D_loss_curr.backward()
        opt.step()
        
        opt.zero_grad()
        G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = model.generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
        G_loss_curr.backward()
        opt.step()    
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
            print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
            print()



def create_test_mask_ls(all_mask_ls, test_ratio = 0.2):
    test_mask1 = []
    test_mask2 = []
    for mask in all_mask_ls:
        test_train_mask = mask.clone()
        test_train_mask2 = mask.clone()
        
        mask_1_index = torch.nonzero(mask == 1)
        
        rand_index = torch.randperm(len(mask_1_index))

        select_rand_index = rand_index[int(len(rand_index)*test_ratio):]
        select_rand_index2 = rand_index[0:int(len(rand_index)*test_ratio)]

        mask_rand_1_index = mask_1_index[select_rand_index]
        mask_rand_1_index2 = mask_1_index[select_rand_index2]
        
        
        test_train_mask[mask_rand_1_index[:,0],mask_rand_1_index[:,1]] = 0
        
        test_train_mask2[mask_rand_1_index2[:,0],mask_rand_1_index2[:,1]] = 0
        
        
        assert (test_train_mask + test_train_mask2 - mask).max() == 0
        assert (test_train_mask + test_train_mask2 - mask).min() == 0
        
        test_mask1.append(mask)
        test_mask2.append(test_train_mask2)
    return test_mask1, test_mask2

def load_data(args, task_output_folder_name, prefix, data_min=None, data_max=None):
    # train_X = torch.from_numpy(load_objs(os.path.join(args.input_dir, "train_X")))
    # train_Y = torch.from_numpy(load_objs(os.path.join(args.input_dir, "train_Y")))
    # test_X = torch.from_numpy(load_objs(os.path.join(args.input_dir, "test_X")))
    # test_Y = torch.from_numpy(load_objs(os.path.join(args.input_dir, "test_Y")))
    # return train_X, train_Y, test_X, test_Y
    # if args.imputed and prefix == "train":
    #     feature_file = os.path.join(task_output_folder_name, prefix + "_X_imputed")
    #     mask_file = os.path.join(task_output_folder_name, prefix + "_mask_imputed")
    #     tp_file = os.path.join(task_output_folder_name, prefix + "_tp_imputed")
    # else:
    feature_file = os.path.join(task_output_folder_name, prefix + "_X")
    mask_file = os.path.join(task_output_folder_name, prefix + "_mask")
    tp_file = os.path.join(task_output_folder_name, prefix + "_tp")

    all_df_ls = load_objs(feature_file)
    all_tp_ls = load_objs(tp_file)
    all_mask_ls = load_objs(mask_file)

    all_df_ls = [torch.from_numpy(df) for df in all_df_ls]
    all_tp_ls = [torch.from_numpy(df) for df in all_tp_ls]
    all_mask_ls = [torch.from_numpy(df) for df in all_mask_ls]
    all_test_mask_ls = None
    if not args.classify_task and not prefix == "train":
        all_mask_ls, all_test_mask_ls = create_test_mask_ls(all_mask_ls)
        
    if args.imputed:
        label_file = os.path.join(task_output_folder_name, prefix + "_Y_imputed")
    else:
        label_file = os.path.join(task_output_folder_name, prefix + "_Y")

    labels_mat = load_objs(label_file)
    labels_mat = torch.from_numpy(labels_mat)
    # visit_tensor_ls, mask_ls, time_step_ls, label_tensor_ls, data_min, data_max
    if data_min is None or data_max is None:
        data_min, data_max = get_data_min_max(all_df_ls, all_mask_ls)
    dataset = mTan_dataset(all_df_ls, all_mask_ls, all_tp_ls, labels_mat, data_min, data_max, all_test_mask_ls)
    return dataset, data_min, data_max
    
    

# def main(args):
#     train_X, train_Y, test_X, test_Y = load_data(args)
#     train_X = train_X.reshape((train_X.shape[0], -1, 42))
#     test_X = test_X.reshape((test_X.shape[0], -1, 42))
#     Dim = train_X.shape[1]
#     H_Dim1, H_Dim2 = Dim, Dim
#     model = Gain(Dim, H_Dim1, H_Dim2)

#     model = model.cuda()

#     optimizer = torch.optim.Adam(model.parameters())

#     train(train_X, model, optimizer, Dim, args.p_hint)

def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def compute_losses(observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, model_config, device):
    # observed_data, observed_mask \
    #     = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = model_config["std"]  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if model_config["norm"]:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl

def mean_squared_error(orig, pred, mask, gt_mask = None):
    if gt_mask is not None:
        mask = mask - gt_mask
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label,_ in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                label = label.long()
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auc = metrics.roc_auc_score(
        true, pred[:, 1]) if not args.classify_pertp else 0.
    return test_loss/pred.shape[0], acc, auc

# def evaluate(dim, rec, dec, test_loader, model_config, num_sample=10, device="cuda"):
#     mse, test_n = 0.0, 0.0
#     with torch.no_grad():
#         for (test_batch,_, observed_test_mask) in test_loader:
#             test_batch = test_batch.to(device)
#             observed_test_mask = observed_test_mask.to(device)
#             observed_data, observed_mask, observed_tp = (
#                 test_batch[:, :, :dim],
#                 test_batch[:, :, dim: 2 * dim],
#                 test_batch[:, :, -1],
#             )
#             # if args.sample_tp and args.sample_tp < 1:
#             #     subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
#             #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
#             # else:
#             subsampled_data, subsampled_tp, subsampled_mask = \
#                 observed_data, observed_tp, observed_mask
#             out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
#             qz0_mean, qz0_logvar = (
#                 out[:, :, : model_config["latent_dim"]],
#                 out[:, :, model_config["latent_dim"]:],
#             )
#             epsilon = torch.randn(
#                 num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
#             ).to(device)
#             z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
#             z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
#             batch, seqlen = observed_tp.size()
#             time_steps = (
#                 observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
#             )
#             pred_x = dec(z0, time_steps)
#             pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
#             pred_x = pred_x.mean(0)
#             mse += mean_squared_error(observed_data, pred_x, observed_test_mask) * batch
#             test_n += batch
#     return mse / test_n

def evaluate(dim, data_min, data_max, model, test_loader, all_estimated_attr_range_mappings, device="cuda"):
    mse, test_n = 0.0, 0.0
    model.eval()
    total_rule_violation_count = 0
    pred_x_ls = []
    with torch.no_grad():
        for (test_batch,observed_test_mask) in tqdm(test_loader):
            test_batch = test_batch.to(device)
            observed_test_mask = observed_test_mask.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            # if args.sample_tp and args.sample_tp < 1:
            #     subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
            #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            # else:
            subsampled_data, subsampled_tp, subsampled_mask = \
                observed_data, observed_tp, observed_mask
            
            pred_x = model.evaluate(subsampled_data, subsampled_tp, subsampled_mask, observed_test_mask)
            pred_x_ls.append(pred_x)
            if all_estimated_attr_range_mappings is not None:
                total_rule_violation_count += evaluate_rule_violation_count(pred_x, all_estimated_attr_range_mappings, data_min, data_max)
            # _, pred_x = model(subsampled_data, subsampled_tp, subsampled_mask)
            # out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            # qz0_mean, qz0_logvar = (
            #     out[:, :, : model_config["latent_dim"]],
            #     out[:, :, model_config["latent_dim"]:],
            # )
            # epsilon = torch.randn(
            #     num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            # ).to(device)
            # z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            # z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            # batch, seqlen = observed_tp.size()
            # time_steps = (
            #     observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            # )
            # pred_x = dec(z0, time_steps)
            # pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            # pred_x = pred_x.mean(0)
            batch, _ = observed_tp.size()
            mse += mean_squared_error(observed_data.detach().cpu(), pred_x.detach().cpu().squeeze(0), subsampled_mask.detach().cpu(),  observed_test_mask.detach().cpu()) * batch
            # mse += mean_squared_error(observed_data.detach().cpu(), pred_x.detach().cpu().squeeze(0), subsampled_mask.detach().cpu()) * batch
            test_n += batch
    # print("total rule violation count::", total_rule_violation_count)
    model.train()
    return mse / test_n, pred_x_ls, total_rule_violation_count

# def train_imputation(args, model_config, train_loader, val_loader,test_loader,  device, dim, optimizer, criterion, rec, dec, data_min, data_max, all_estimated_attr_range_mappings):
#     best_valid_mse = float('inf')
#     data_min = data_min.to(device)
#     data_max = data_max.to(device)
#     for itr in range(1, model_config["epochs"] + 1):
#         train_loss = 0
#         train_n = 0
#         avg_reconst, avg_kl, mse = 0, 0, 0
#         if model_config["kl"]:
#             wait_until_kl_inc = 10
#             if itr < wait_until_kl_inc:
#                 kl_coef = 0.
#             else:
#                 kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
#         else:
#             kl_coef = 1
            
#         kargs = {"kl_coef": kl_coef}

#         for (train_batch,_,_) in tqdm(train_loader):
#             train_batch = train_batch.to(device)
#             batch_len = train_batch.shape[0]
#             observed_data = train_batch[:, :, :dim]
#             observed_mask = train_batch[:, :, dim:2 * dim]
#             observed_tp = train_batch[:, :, -1]
#             # if args.sample_tp and args.sample_tp < 1:
#             #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
#             #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
#             # else:
#             subsampled_data, subsampled_tp, subsampled_mask = \
#                 observed_data, observed_tp, observed_mask
#             out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
#             qz0_mean = out[:, :, :model_config["latent_dim"]]
#             qz0_logvar = out[:, :, model_config["latent_dim"]:]
#             # epsilon = torch.randn(qz0_mean.size()).to(device)
#             epsilon = torch.randn(
#                 model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
#             ).to(device)
#             z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
#             z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
#             pred_x = dec(
#                 z0,
#                 observed_tp[None, :, :].repeat(model_config["k_iwae"], 1, 1).view(-1, observed_tp.shape[1])
#             )
#             # nsample, batch, seqlen, dim
#             pred_x = pred_x.view(model_config["k_iwae"], batch_len, pred_x.shape[1], pred_x.shape[2])
#             # compute loss
#             # loss = torch.sum(torch.abs(pred_x))
#             logpx, analytic_kl = compute_losses(
#                 observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, model_config, device)
#             loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(model_config["k_iwae"]))
            
            
#             # rule_loss = eval_loss_on_rules(pred_x, observed_mask, data_min, data_max, all_estimated_attr_range_mappings)
            
#             # loss += args.beta*rule_loss
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             del pred_x, loss, subsampled_data, subsampled_tp, subsampled_mask, observed_data, observed_tp, observed_mask, train_batch, out, epsilon
#             # train_loss += loss.item() * batch_len
#             # train_n += batch_len
#             # avg_reconst += torch.mean(logpx.detach().cpu()) * batch_len
#             # avg_kl += torch.mean(analytic_kl.detach().cpu()) * batch_len
#             # mse += mean_squared_error(
#             #     observed_data.cpu(), pred_x.detach().cpu().mean(0), observed_mask.cpu()) * batch_len

#         valid_mse = evaluate(dim, rec, dec, val_loader, model_config, 1)
#         test_mse = evaluate(dim, rec, dec, test_loader, model_config, 1)
#         print('Mean Squared Error',valid_mse, test_mse)
#         if valid_mse <= best_valid_mse:
#             best_valid_mse = valid_mse
#             best_test_mse = test_mse
#             rec_state_dict = rec.state_dict()
#             dec_state_dict = dec.state_dict()
#             rec_file_path = os.path.join(args.log_path, "model_rec_best")
#             torch.save(rec_state_dict, rec_file_path)
#             dec_file_path = os.path.join(args.log_path, "model_dec_best")
#             torch.save(dec_state_dict, dec_file_path)
        
#     print('Best Mean Squared Error',best_test_mse)


def train_imputation(args, train_config, model_config, train_loader, val_loader,test_loader,  device, dim, optimizer, model , data_min, data_max, all_estimated_attr_range_mappings, lr_scheduler=None):
    best_valid_mse = float('inf')
    data_min = data_min.to(device)
    data_max = data_max.to(device)
    valid_mse, train_imputed_x,_ = evaluate(dim, data_min, data_max, model, val_loader, all_estimated_attr_range_mappings, device)
    test_mse, valid_imputed_x,_ = evaluate(dim, data_min, data_max,model, test_loader, all_estimated_attr_range_mappings, device)
    print('Mean Squared Error',valid_mse, test_mse)
    for itr in range(1, train_config["epochs"] + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        kargs={}
        if args.model_type == "mTan":
            if train_config["kl"]:
                wait_until_kl_inc = 10
                if itr < wait_until_kl_inc:
                    kl_coef = 0.
                else:
                    kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
            else:
                kl_coef = 1
            kargs["kl_coef"] = kl_coef
        
        if args.model_type == "saits":
            kargs["imputation_loss_weight"] = train_config["imputation_loss_weight"]
            kargs["reconstruction_loss_weight"] = train_config["reconstruction_loss_weight"]


        for (train_batch,_) in tqdm(train_loader):
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            # if args.sample_tp and args.sample_tp < 1:
            #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
            #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            # else:
            subsampled_data, subsampled_tp, subsampled_mask = \
                observed_data, observed_tp, observed_mask
                
            # out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            # qz0_mean = out[:, :, :model_config["latent_dim"]]
            # qz0_logvar = out[:, :, model_config["latent_dim"]:]
            # # epsilon = torch.randn(qz0_mean.size()).to(device)
            # epsilon = torch.randn(
            #     model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            # ).to(device)
            # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            # z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            # pred_x = dec(
            #     z0,
            #     subsampled_tp[None, :, :].repeat(model_config["k_iwae"], 1, 1).view(-1, subsampled_tp.shape[1])
            # )
            # # nsample, batch, seqlen, dim
            # pred_x = pred_x.view(model_config["k_iwae"], subsampled_data.shape[0], pred_x.shape[1], pred_x.shape[2])
            # # compute loss
            # logpx, analytic_kl = compute_losses(
            #     subsampled_data, subsampled_mask, qz0_mean, qz0_logvar, pred_x, model_config, device)
            loss = model(subsampled_data, subsampled_tp, subsampled_mask, **kargs)
            # loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(model_config["k_iwae"]))
            # out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            # qz0_mean = out[:, :, :model_config["latent_dim"]]
            # qz0_logvar = out[:, :, model_config["latent_dim"]:]
            # # epsilon = torch.randn(qz0_mean.size()).to(device)
            # epsilon = torch.randn(
            #     model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            # ).to(device)
            # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            # z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            # pred_x = dec(
            #     z0,
            #     observed_tp[None, :, :].repeat(model_config["k_iwae"], 1, 1).view(-1, observed_tp.shape[1])
            # )
            # # nsample, batch, seqlen, dim
            # pred_x = pred_x.view(model_config["k_iwae"], batch_len, pred_x.shape[1], pred_x.shape[2])
            # # compute loss
            # logpx, analytic_kl = compute_losses(
            #     dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            # loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(model_config["k_iwae"]))
            # rule_loss = eval_loss_on_rules(pred_x, observed_mask, data_min, data_max, all_estimated_attr_range_mappings)
            
            # loss += args.beta*rule_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            # avg_reconst += torch.mean(logpx) * batch_len
            # avg_kl += torch.mean(analytic_kl) * batch_len
            # mse += mean_squared_error(
            #     observed_data.cpu(), pred_x.detach().cpu().squeeze(0), observed_mask.cpu()) * batch_len
            del loss, observed_data, observed_tp, observed_mask, train_batch, subsampled_data, subsampled_tp, subsampled_mask

        if lr_scheduler is not None:
            lr_scheduler.step()


        valid_mse,_,_ = evaluate(dim,data_min, data_max,  model, val_loader, all_estimated_attr_range_mappings, device)
        test_mse,_,_ = evaluate(dim, data_min, data_max, model, test_loader, all_estimated_attr_range_mappings, device)
        print('Mean Squared Error',valid_mse, test_mse)
        if valid_mse <= best_valid_mse:
            best_valid_mse = valid_mse
            best_test_mse = test_mse
            # rec_state_dict = rec.state_dict()
            # dec_state_dict = dec.state_dict()
            model_dict = model.state_dict()
            model_file_path = os.path.join(args.full_log_path, "model_best")
            torch.save(model_dict, model_file_path)
            # dec_file_path = os.path.join(args.log_path, "model_dec_best")
            # torch.save(dec_state_dict, dec_file_path)
        
    print('Best Mean Squared Error',best_test_mse)

# def train_imputation_tta(args, model_config, test_loader,  device, dim, optimizer, criterion, rec, dec, data_min, data_max, all_estimated_attr_range_mappings):
#     best_valid_mse = float('inf')
#     data_min = data_min.to(device)
#     data_max = data_max.to(device)
#     test_mse = evaluate(dim, rec, dec, test_loader, device)
#     print('Mean Squared Error',test_mse)
#     for itr in range(1, args.epochs + 1):
#         train_loss = 0
#         train_n = 0
#         avg_reconst, avg_kl, mse = 0, 0, 0
#         if model_config["kl"]:
#             wait_until_kl_inc = 10
#             if itr < wait_until_kl_inc:
#                 kl_coef = 0.
#             else:
#                 kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
#         else:
#             kl_coef = 1

#         for (train_batch,_,observed_test_mask) in tqdm(test_loader):
#             train_batch = train_batch.to(device)
#             batch_len = train_batch.shape[0]
#             observed_data = train_batch[:, :, :dim]
#             observed_mask = train_batch[:, :, dim:2 * dim]
#             observed_tp = train_batch[:, :, -1]
#             # if args.sample_tp and args.sample_tp < 1:
#             #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
#             #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
#             # else:
#             subsampled_data, subsampled_tp, subsampled_mask = \
#                 observed_data, observed_tp, observed_mask
#             out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
#             qz0_mean = out[:, :, :model_config["latent_dim"]]
#             qz0_logvar = out[:, :, model_config["latent_dim"]:]
#             # epsilon = torch.randn(qz0_mean.size()).to(device)
#             epsilon = torch.randn(
#                 model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
#             ).to(device)
#             z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
#             z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
#             pred_x = dec(
#                 z0,
#                 observed_tp[None, :, :].repeat(model_config["k_iwae"], 1, 1).view(-1, observed_tp.shape[1])
#             )
#             # nsample, batch, seqlen, dim
#             pred_x = pred_x.view(model_config["k_iwae"], batch_len, pred_x.shape[1], pred_x.shape[2])
#             # compute loss
#             # logpx, analytic_kl = compute_losses(
#                 # dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
#             # loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(model_config["k_iwae"]))
#             rule_loss = eval_loss_on_rules(pred_x, observed_test_mask, data_min, data_max, all_estimated_attr_range_mappings)
            
#             loss = model_config["beta"]*rule_loss
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * batch_len
#             train_n += batch_len
#             # avg_reconst += torch.mean(logpx) * batch_len
#             # avg_kl += torch.mean(analytic_kl) * batch_len
#             mse += mean_squared_error(
#                 observed_data, pred_x.mean(0), observed_mask) * batch_len
            
#             del subsampled_data, subsampled_tp, subsampled_mask, rule_loss, loss, out, qz0_mean, qz0_logvar, pred_x

#         # valid_mse = evaluate(dim, rec, dec, val_loader, model_config, 1)
#         test_mse = evaluate(dim, rec, dec, test_loader, device)
#         print('Mean Squared Error',test_mse)
        # if valid_mse <= best_valid_mse:
        #     best_valid_mse = valid_mse
        #     best_test_mse = test_mse
            
        
    # print('Best Mean Squared Error',best_test_mse)

def evaluate_rule_violation_count(pred_x, all_estimated_attr_range_mappings, data_min, data_max):
    total_violated_count = 0
    for idx in range(len(all_estimated_attr_range_mappings)):
        estimated_range = all_estimated_attr_range_mappings[idx]
        rescaled_pred_x = pred_x[:,:,idx]*(data_max[idx] - data_min[idx]) + data_min[idx]
        # total_violated_count += (torch.sum(pred_x[:,:,idx] > estimated_range[1]) + torch.sum(pred_x[:,:,idx] < estimated_range[0]))
        total_violated_count += (torch.sum(rescaled_pred_x > estimated_range[1]) + torch.sum(rescaled_pred_x < estimated_range[0]))

    return total_violated_count

def save_objs(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def train_imputation_tta(args, train_config, model_config, test_loader, test_eval_loader, device, dim, optimizer, model, data_min, data_max, all_estimated_attr_range_mappings):
    best_valid_mse = float('inf')
    data_min = data_min.to(device)
    data_max = data_max.to(device)
    init_test_mse, test_imputed_x, init_violation_count = evaluate(dim, data_min, data_max, model, test_eval_loader, all_estimated_attr_range_mappings, device)
    
    save_objs(test_imputed_x, os.path.join(args.log_path, "test_imputed_x"))
    save_objs(test_eval_loader, os.path.join(args.log_path, "test_eval_loader"))
    
    print('Mean Squared Error',init_test_mse)
    print("total rule violation count::", init_violation_count)
    best_test_mse = np.inf
    best_violation_count = np.inf
    for itr in range(1, train_config["epochs"] + 1):
        # train_loss = 0
        # train_n = 0
        # avg_reconst, avg_kl, mse = 0, 0, 0
        # if train_config["kl"]:
        #     wait_until_kl_inc = 10
        #     if itr < wait_until_kl_inc:
        #         kl_coef = 0.
        #     else:
        #         kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        # else:
        #     kl_coef = 1

        for (train_batch,observed_test_mask) in tqdm(test_loader):
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            # if args.sample_tp and args.sample_tp < 1:
            #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
            #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            # else:
            subsampled_data, subsampled_tp, subsampled_mask = \
                observed_data, observed_tp, observed_mask
            observed_test_mask = observed_test_mask.to(device)
            # out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            # qz0_mean = out[:, :, :model_config["latent_dim"]]
            # qz0_logvar = out[:, :, model_config["latent_dim"]:]
            # # epsilon = torch.randn(qz0_mean.size()).to(device)
            # epsilon = torch.randn(
            #     model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            # ).to(device)
            # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            # z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            # pred_x = dec(
            #     z0,
            #     observed_tp[None, :, :].repeat(model_config["k_iwae"], 1, 1).view(-1, observed_tp.shape[1])
            # )
            # # nsample, batch, seqlen, dim
            # pred_x = pred_x.view(model_config["k_iwae"], batch_len, pred_x.shape[1], pred_x.shape[2])

            if not args.model_type == "csdi":
                    pred_x = model.evaluate(subsampled_data, subsampled_tp, subsampled_mask, observed_test_mask)
            else:
                pred_x = None
                for t in range(model.num_steps - 1, -1, -1):
                    if pred_x is not None:
                        pred_x = pred_x.detach()
                    pred_x = model.impute_main(subsampled_data, subsampled_tp, subsampled_mask, observed_test_mask, imputed_samples=pred_x, t=t)

            if args.tta_method == 'rule':
                if not args.model_type == "csdi":
                    # pred_x = model.evaluate(subsampled_data, subsampled_tp, subsampled_mask, observed_test_mask)
                # # compute loss
                # logpx, analytic_kl = compute_losses(
                    # dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
                # loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(model_config["k_iwae"]))
                    rule_loss = eval_loss_on_rules(pred_x, torch.ones_like(pred_x), data_min, data_max, all_estimated_attr_range_mappings)
                    
                    loss = train_config["beta"]*rule_loss
                    
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                else:
                    # pred_x = None
                    # for t in range(model.num_steps - 1, -1, -1):
                    #     if pred_x is not None:
                    #         pred_x = pred_x.detach()
                    #     pred_x = model.impute_main(subsampled_data, subsampled_tp, subsampled_mask, observed_test_mask, imputed_samples=pred_x, t=t)
                    rule_loss = eval_loss_on_rules(pred_x, torch.ones_like(pred_x), data_min, data_max, all_estimated_attr_range_mappings)
                    
                    loss = train_config["beta"]*rule_loss
                    
            elif args.tta_method == "cpl":
                
                
                # outputs = torch.sigmoid(net(inputs_num, inputs_cat))
                # if len(pred_x) > 0:
                loss = conjugate_pl(pred_x.view(-1, pred_x.shape[-1]), num_classes=pred_x.shape[-1], input_probs=True)
                loss = train_config["beta"]*loss
                # losses['cpl_bbox_loss'] = loss/1000
                # else:
                #     synth_data = torch.rand([1,10], requires_grad=True)
                #     losses['cpl_bbox_loss'] = torch.mean(synth_data)
            elif args.tta_method == "rpl":
                # outputs = torch.sigmoid(net(inputs_num, inputs_cat))
                # if len(det_bboxes) > 0:
                loss = robust_pl(pred_x, input_probs=False)
                loss = train_config["beta"]*loss*0.0001
                # losses['rpl_bbox_loss'] = loss/1000 + 1
                # else:
                #     synth_data = torch.rand([1,10], requires_grad=True)
                #     losses['rpl_bbox_loss'] = torch.mean(synth_data)
            # elif args.tta_method == 'memo':
            #     return det_bboxes[:,0:4]
            elif args.tta_method == "norm":
                synth_data = torch.rand([1,10], requires_grad=True)
                loss = torch.mean(synth_data)
                loss = train_config["beta"]*loss
                
            elif args.tta_method == "tent":
                max_norm = torch.max(pred_x**2).item()

                # loss = -0.5*torch.norm(bbox/max_norm, p="fro")
                loss = -0.5*torch.mean(pred_x**2/max_norm) + 1
                loss = train_config["beta"]*loss
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_loss += loss.item() * batch_len
            # train_n += batch_len
            # # avg_reconst += torch.mean(logpx) * batch_len
            # # avg_kl += torch.mean(analytic_kl) * batch_len
            # mse += mean_squared_error(
            #     observed_data.cpu(), pred_x.detach().cpu(), observed_mask) * batch_len
            
            del subsampled_data, subsampled_tp, subsampled_mask, loss, pred_x

        # valid_mse = evaluate(dim, rec, dec, val_loader, model_config, 1)
        test_mse, _, violation_count = evaluate(dim,data_min, data_max,  model, test_eval_loader, all_estimated_attr_range_mappings,  device)
        
        if test_mse <= best_test_mse :
            best_test_mse = test_mse
            best_violation_count = violation_count
        
        print('Mean Squared Error:',test_mse, init_test_mse)
        print("Reduction of Mean Squared Error: %f"%((init_test_mse.item() - test_mse.item())/init_test_mse.item()))
        print("total rule violation count::", violation_count, init_violation_count)
        print("Reduction of the rule violation count::%f"%((init_violation_count.item()-violation_count.item())/init_violation_count.item()))
        # if valid_mse <= best_valid_mse:
        #     best_valid_mse = valid_mse
        #     best_test_mse = test_mse
            
    print('Mean Squared Error:',best_test_mse, init_test_mse)
    print("Reduction of Mean Squared Error: %f"%((init_test_mse.item() - best_test_mse.item())/init_test_mse.item()))
    print("total rule violation count::", best_violation_count, init_violation_count)
    print("Reduction of the rule violation count::%f"%((init_violation_count.item()-best_violation_count.item())/init_violation_count.item()))
    # print('Best Mean Squared Error',best_test_mse)

def eval_loss_on_rules(pred_x, mask, att_min, att_max, all_estimated_attr_range_mappings):
    rule_loss = 0
    pred_x = pred_x.squeeze(0)
    renomalized_pred_x = renormalize_masked_data(pred_x, mask, att_min, att_max)
    for idx in range(renomalized_pred_x.shape[-1]):
        curr_feat_vals = renomalized_pred_x[:,:,idx].reshape(-1)
        low_bound, upper_bound = all_estimated_attr_range_mappings[idx]
        curr_feat_loss = (curr_feat_vals - low_bound)*(curr_feat_vals - upper_bound)
        
        clamped_curr_feat_loss = torch.clamp(curr_feat_loss, min=0.0, max=1.0)
        
        rule_loss += torch.mean(clamped_curr_feat_loss)
    rule_loss = rule_loss/renomalized_pred_x.shape[-1]
    return rule_loss
    
        

# def mTan_train_classification(args,data_min, data_max, train_loader, val_loader,test_loader,  device, dim, optimizer, criterion, rec, dec, classifier, all_estimated_attr_range_mappings):
#     best_val_loss = float('inf')
#     val_loss, val_acc, val_auc = evaluate_classifier(
#         rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
    
#     test_loss, test_acc, test_auc = evaluate_classifier(
#         rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
    
#     print('val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
#               .format(val_loss, val_acc, test_acc, test_auc))
#     data_min = data_min.to(device)
#     data_max = data_max.to(device)
#     for itr in range(1, args.epochs + 1):
#         train_recon_loss, train_ce_loss = 0, 0
#         mse = 0
#         train_n = 0
#         train_acc = 0
#         #avg_reconst, avg_kl, mse = 0, 0, 0
#         if args.kl:
#             wait_until_kl_inc = 10
#             if itr < wait_until_kl_inc:
#                 kl_coef = 0.
#             else:
#                 kl_coef = (1-0.99** (itr - wait_until_kl_inc))
#         else:
#             kl_coef = 1
#         # start_time = time.time()
#         for train_batch, label,_ in tqdm(train_loader):
#             train_batch, label = train_batch.to(device), label.to(device)
#             batch_len  = train_batch.shape[0]
#             observed_data, observed_mask, observed_tp \
#                 = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
#             out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
#             qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
#             epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
#             z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
#             z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
#             pred_y = classifier(z0)
#             pred_x = dec(
#                 z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
#             pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
#             # compute loss
#             logpx, analytic_kl = compute_losses(
#                 dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
#             recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
#             label = label.long().unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
#             ce_loss = criterion(pred_y, label)
#             loss = recon_loss + args.alpha*ce_loss
#             # pred_x, mask, att_min, att_max, all_estimated_attr_range_mappings
#             rule_loss = eval_loss_on_rules(pred_x, observed_mask, data_min, data_max, all_estimated_attr_range_mappings)
            
#             loss += args.beta*rule_loss
            
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_ce_loss += ce_loss.item() * batch_len
#             train_recon_loss += recon_loss.item() * batch_len
#             train_acc += (pred_y.argmax(1) == label).sum().item()/args.k_iwae
#             train_n += batch_len
#             mse += mean_squared_error(observed_data, pred_x.mean(0), 
#                                       observed_mask) * batch_len
#         # total_time += time.time() - start_time
#         val_loss, val_acc, val_auc = evaluate_classifier(
#             rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        
#         test_loss, test_acc, test_auc = evaluate_classifier(
#             rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
#         print('Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
#               .format(itr, train_recon_loss/train_n, train_ce_loss/train_n, 
#                       train_acc/train_n, mse/train_n, val_loss, val_acc, test_acc, test_auc))
        
#         if val_loss <= best_val_loss:
#             best_val_loss = min(best_val_loss, val_loss)
#             rec_state_dict = rec.state_dict()
#             dec_state_dict = dec.state_dict()
#             classifier_state_dict = classifier.state_dict()
#             optimizer_state_dict = optimizer.state_dict()
#             best_test_loss = test_loss
#             best_test_acc = test_acc
#             best_test_auc = test_auc
#             if not os.path.exists(args.log_path):
#                 os.makedirs(args.log_path)
            
#             rec_file_path = os.path.join(args.log_path, "model_rec_best")
#             torch.save(rec_state_dict, rec_file_path)
#             dec_file_path = os.path.join(args.log_path, "model_dec_best")
#             torch.save(dec_state_dict, dec_file_path)
#             classifier_file_path = os.path.join(args.log_path, "model_classifier_best")
#             torch.save(classifier_state_dict, classifier_file_path)
        
#     print('best test_acc: {:.4f}, best test_auc: {:.4f}'
#               .format(best_test_acc, best_test_auc))
#         # if itr % 100 == 0 and args.save:
#         #     torch.save({
#         #         'args': args,
#         #         'epoch': itr,
#         #         'rec_state_dict': rec_state_dict,
#         #         'dec_state_dict': dec_state_dict,
#         #         'optimizer_state_dict': optimizer_state_dict,
#         #         'classifier_state_dict': classifier_state_dict,
#         #         'loss': -loss,
#         #     }, args.dataset + '_' + 
#         #         args.enc + '_' + 
#         #         args.dec + '_' + 
#         #         str(experiment_id) +
#         #         '.h5')


def retrieve_attributes_ranges(folder):
    
    

    all_estimated_dist_file = os.path.join(folder, "all_attr_range_mappings")

    all_estimated_attr_range_mappings = load_objs(all_estimated_dist_file)
    
    
    if not type(all_estimated_attr_range_mappings) is list:
        attr_range_ls = []
        
        for key in attr_ls:
            attr_range_ls.append(all_estimated_attr_range_mappings[key])
    
        return attr_range_ls
    else:
        return all_estimated_attr_range_mappings


def main_saits(test_time_config, train_config, model_config, train_dataset, train_loader, valid_loader, test_loader,test_eval_loader, device, data_min, data_max, all_estimated_attr_range_mappings):
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    time_len = train_dataset.__getitem__(1)[0].shape[-2]
    # model = CSDI_Physio(model_config, device, target_dim=dim).to(device)
    model_config["device"] = device
    model_config["d_feature"] = dim
    model_config["d_time"] = time_len
    model = SAITS(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=1e-6)
    # if foldername != "":
    #     output_path = foldername + "/model.pth"

    # p1 = int(0.75 * train_config["epochs"])
    # p2 = int(0.9 * train_config["epochs"])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[p1, p2], gamma=0.1
    # )
    
    if args.cache_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.cache_path, "model_best")))
    
    if args.do_train:
        train_imputation(args, train_config, model_config, train_loader, valid_loader, test_loader,  device, dim, optimizer, model , data_min, data_max, all_estimated_attr_range_mappings)
    else:
        # when we do test time adaptations, we follow the state-of-the-art to only fine-tune the normalization layers. The following code is for only retrieving the normalization layers and create an optimizer upon them
        model = tent.configure_model(model)
        params,_ = tent.collect_params(model)
        # params = (list(rec_params) + list(dec_params) + list(classifier.parameters()))
        optimizer = torch.optim.Adam(params, lr=train_config["lr"], weight_decay=1e-6)
        train_imputation_tta(args, test_time_config, model_config, test_loader, test_eval_loader, device, dim, optimizer, model, data_min, data_max, all_estimated_attr_range_mappings)


def main_csdi(test_time_config, train_config, model_config, train_dataset, train_loader, valid_loader, test_loader, test_eval_loader, device, data_min, data_max, all_estimated_attr_range_mappings):
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    model = CSDI_Physio(model_config, device, target_dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=1e-6)
    # if foldername != "":
    #     output_path = foldername + "/model.pth"

    p1 = int(0.75 * train_config["epochs"])
    p2 = int(0.9 * train_config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    
    if args.cache_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.cache_path, "model_best")))
    
    if args.do_train:
        train_imputation(args, train_config, model_config, train_loader, valid_loader, test_loader,  device, dim, optimizer, model , data_min, data_max, all_estimated_attr_range_mappings, lr_scheduler=lr_scheduler)
    else:
        # when we do test time adaptations, we follow the state-of-the-art to only fine-tune the normalization layers. The following code is for only retrieving the normalization layers and create an optimizer upon them
        model = tent.configure_model(model)
        params,_ = tent.collect_params(model)
        # params = (list(rec_params) + list(dec_params) + list(classifier.parameters()))
        optimizer = torch.optim.Adam(params, lr=train_config["lr"], weight_decay=1e-6)
        train_imputation_tta(args, test_time_config, model_config, test_loader, test_eval_loader, device, dim, optimizer, model, data_min, data_max, all_estimated_attr_range_mappings)
        
    #     # rec = tent.configure_model(rec)
    #     # rec_params, param_names = tent.collect_params(rec)
    #     # dec = tent.configure_model(dec)
    #     # dec_params, param_names = tent.collect_params(dec)
    #     # params = (list(rec_params) + list(dec_params) + list(classifier.parameters()))
    #     # optimizer = torch.optim.Adam(params, lr=lr)
    #     mTan_train_imputation_tta(args, model_config, test_loader,  device, dim, optimizer, criterion, rec, dec, data_min, data_max, all_estimated_attr_range_mappings)


    # best_valid_loss = 1e10
    # for epoch_no in range(train_config["epochs"]):
    #     avg_loss = 0
    #     model.train()
    #     with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
    #         for batch_no, train_batch in enumerate(it, start=1):
    #             train_batch = train_batch.to(device)
    #             batch_len = train_batch.shape[0]
    #             observed_data = train_batch[:, :, :dim]
    #             observed_mask = train_batch[:, :, dim:2 * dim]
    #             observed_tp = train_batch[:, :, -1]
    #             # if args.sample_tp and args.sample_tp < 1:
    #             #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
    #             #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
    #             # else:
    #             subsampled_data, subsampled_tp, subsampled_mask = \
    #                 observed_data, observed_tp, observed_mask
    #             optimizer.zero_grad()

    #             loss = model(subsampled_data, subsampled_tp, subsampled_mask)
    #             loss.backward()
    #             avg_loss += loss.item()
    #             optimizer.step()
    #             it.set_postfix(
    #                 ordered_dict={
    #                     "avg_epoch_loss": avg_loss / batch_no,
    #                     "epoch": epoch_no,
    #                 },
    #                 refresh=False,
    #             )
    #         lr_scheduler.step()
    #     if valid_loader is not None:# and (epoch_no + 1) % valid_epoch_interval == 0:
    #         model.eval()
    #         avg_loss_valid = 0
    #         with torch.no_grad():
    #             with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
    #                 for batch_no, valid_batch in enumerate(it, start=1):
    #                     loss = model(valid_batch, is_train=0)
    #                     avg_loss_valid += loss.item()
    #                     it.set_postfix(
    #                         ordered_dict={
    #                             "valid_avg_epoch_loss": avg_loss_valid / batch_no,
    #                             "epoch": epoch_no,
    #                         },
    #                         refresh=False,
    #                     )
    #         if best_valid_loss > avg_loss_valid:
    #             best_valid_loss = avg_loss_valid
    #             print(
    #                 "\n best loss is updated to ",
    #                 avg_loss_valid / batch_no,
    #                 "at",
    #                 epoch_no,
    #             )

    # if foldername != "":
    #     torch.save(model.state_dict(), output_path)


def main_mTan(test_time_config, train_dataset, train_loader, train_config, model_config, device, val_loader, test_loader, test_eval_loader, all_estimated_attr_range_mappings, data_min, data_max):
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    # static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]

    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., model_config["num_ref_points"]), model_config["latent_dim"], model_config["rec_hidden"], embed_time=128, learn_emb=model_config["learn_emb"], num_heads=model_config["enc_num_heads"], device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., model_config["num_ref_points"]), model_config["latent_dim"], model_config["gen_hidden"], embed_time=128, learn_emb=model_config["learn_emb"], num_heads=model_config["dec_num_heads"], device=device)
    lr = train_config["lr"]
    classifier = create_classifier(model_config["latent_dim"], 20)

    model = mtan_model_full(rec, dec, model_config, device)

    if args.cache_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.cache_path, "model_best")))
        # dec.load_state_dict(torch.load(os.path.join(args.cache_path, "model_dec_best")))
        # classifier.load_state_dict(torch.load(os.path.join(args.cache_path, "model_classifier_best")))

    model.train()
    # dec.train()
    # classifier.train()
    model = model.to(device)
    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)

    # params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    params = list(model.parameters())
    # print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    # args.k_iwae = 1

    # args.norm = True

    # args.std = 0.01

    # # model = mTan_model(rec, dec, classifier, latent_dim, k_iwae, device)

    # # args.alpha=10

    # args.kl_coef = 1
    
    # if args.classify_task:
    #     mTan_train_classification(args, data_min, data_max, train_loader, val_loader,test_loader,  device, dim, optimizer, criterion, rec, dec, classifier, all_estimated_attr_range_mappings)
    # else:
        # args, test_loader,  device, dim, optimizer, criterion, rec, dec, data_min, data_max, all_estimated_attr_range_mappings
    if args.do_train:
        # args, train_config, model_config, train_loader, val_loader,test_loader,  device, dim, optimizer, model , data_min, data_max, all_estimated_attr_range_mappings, lr_scheduler=None
        train_imputation(args, train_config, model_config, train_loader, val_loader, test_loader,  device, dim, optimizer, model , data_min, data_max, all_estimated_attr_range_mappings)
    else:
        
        # rec = tent.configure_model(rec)
        # rec_params, param_names = tent.collect_params(rec)
        model = tent.configure_model(model)
        _, params = tent.collect_params(model)
        # params = (list(rec_params) + list(dec_params) + list(classifier.parameters()))
        optimizer = torch.optim.Adam(params, lr=lr)
        # train_imputation_tta(args, model_config, test_loader,  device, dim, optimizer, criterion, rec, dec, data_min, data_max, all_estimated_attr_range_mappings)
        train_imputation_tta(args, test_time_config, model_config, test_loader, test_eval_loader, device, dim, optimizer, model, data_min, data_max, all_estimated_attr_range_mappings)




def main(args):
    
    model_config_file_path_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_configs")
    
    # if args.model_type == "mTan":
    yamlfile_name = os.path.join(model_config_file_path_base, "model_config.yaml")
    # elif args.model_type == "csdi":
    #     yamlfile_name = os.path.join(model_config_file_path_base, "csdi_config.yaml")
    with open(yamlfile_name, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        config = config[args.model_type]
        train_config = config["train"]
        model_config = config["model"]
        test_time_config = config["test_time"]
    # task_output_folder_name = os.path.join(args.output, args.task_name)

    # if args.dataset == "mimic3":

    #     train_dataset, data_min, data_max = load_data(args, task_output_folder_name, "train")
    #     valid_dataset, _,_ = load_data(args, task_output_folder_name, "val",data_min, data_max)
    #     test_dataset, _,_ = load_data(args, task_output_folder_name, "test",data_min, data_max)
        
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, collate_fn=mTan_dataset.collate_fn)
    #     val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_config["batch_size"], shuffle=False, collate_fn=mTan_dataset.collate_fn)
    #     if args.do_train:
    #         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, collate_fn=mTan_dataset.collate_fn)
    #         test_eval_loader = None
    #     else:
    #         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=True, collate_fn=mTan_dataset.collate_fn)
    #         test_eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, collate_fn=mTan_dataset.collate_fn)
    
    #     task_output_folder = os.path.join(args.output, args.task_name)
    #     all_estimated_attr_range_mappings = retrieve_attributes_ranges(task_output_folder)
    # elif args.dataset == "physionet":
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.full_log_path = os.path.join(args.log_path, args.model_type)
    if not os.path.exists(args.full_log_path):
        os.makedirs(args.full_log_path)
    if args.cache_path is None:
        args.cache_path = args.full_log_path
    
    train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader, data_max, data_min = get_dataloader(root_data_path=args.input, batch_size=train_config["batch_size"], missing_ratio=args.missing_ratio, do_train=args.do_train)
    data_max = torch.from_numpy(data_max)
    data_min = torch.from_numpy(data_min)
    
    
        
    # task_output_folder = os.path.join(args.output, args.task_name)
    all_estimated_attr_range_mappings = None
    if not args.do_train:
        all_estimated_attr_range_mappings = retrieve_attributes_ranges(args.output)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=True, collate_fn=Physio_Dataset.collate_fn)
    
    test_eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, collate_fn=Physio_Dataset.collate_fn)

    # model_config = Namespace(**model_config)

    

    
    
    # args.latent_dim=20
    # rec_hidden=64
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    
    # train_X = train_X.reshape((train_X.shape[0], -1, 42))
    # test_X = test_X.reshape((test_X.shape[0], -1, 42))
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    
    save_objs(train_loader, os.path.join(args.log_path, "train_loader"))
    save_objs(val_loader, os.path.join(args.log_path, "val_loader"))
    
    if args.model_type == "mTan":
        main_mTan(test_time_config, train_dataset, train_loader, train_config, model_config, device, val_loader, test_loader, test_eval_loader, all_estimated_attr_range_mappings, data_min, data_max)
    
    elif args.model_type == "csdi":
        
        main_csdi(test_time_config, train_config, model_config, train_dataset, train_loader, val_loader, test_loader, test_eval_loader, device, data_min, data_max, all_estimated_attr_range_mappings)
        
    elif args.model_type == "saits":
        main_saits(test_time_config, train_config, model_config, train_dataset, train_loader, val_loader, test_loader, test_eval_loader, device, data_min, data_max, all_estimated_attr_range_mappings)
    
if __name__ == '__main__':


    # mb_size = 128
    # # 2. Missing rate
    # p_miss = 0.2
    # # 3. Hint rate
    # p_hint = 0.9
    # # 4. Loss Hyperparameters
    # alpha = 10
    # # 5. Train Rate
#     # train_rate = 0.8

#     parser = argparse.ArgumentParser(description="parse args")
#     # parser.add_argument('--model', type=str, default='DHMM_cluster', help='model name')
#     # parser.add_argument('--dataset', type=str, default='climate_NY', help='name of dataset')
# #     parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
#     parser.add_argument('--input_dir', type=str, help="std of the initial phi table")
#     parser.add_argument('--imputed', action='store_true', help='specifies what features to extract')
#     # parser.add_argument('--alpha', type=float, default=10, help="std of the initial phi table")
#     # parser.add_argument('--train_rate', type=float, default=0.8, help="std of the initial phi table")
#     # parser.add_argument('--p_hint', type=float, default=0.9, help="std of the initial phi table")
#     # parser.add_argument('--mb_size', type=int, default=128, help="std of the initial phi table")

    # args = parser.parse_args()
    args = parse_args()

    main(args)
    
    
    
    
    
    