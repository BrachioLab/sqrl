import torch
import numpy as np

right_align = True


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max

def renormalize_masked_data(data, mask, att_min, att_max):
    # att_max[att_max == 0.] = 1.
    data = data*(att_max-att_min).view(1,1,-1) + att_min.view(1,1,-1)
    data[mask == 0] = 0
    return data

def variable_time_collate_fn(tt_ls, val_ls, mask_ls, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None, test_mask_ls = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = val_ls[0].shape[1]
    # number of labels
    # N = labels_ls[0].shape[1] if activity else 1
    len_tt = [ex.size(0) for ex in val_ls]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(val_ls), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(val_ls), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(val_ls), maxlen, D]).to(device)
    enc_combined_test_mask = None
    if test_mask_ls is not None:
        enc_combined_test_mask = torch.zeros([len(val_ls), maxlen, D]).to(device)
    # if classify:
    # if activity:
    #     combined_labels = torch.zeros([len(val_ls), maxlen, N]).to(device)
    # else:
    #     combined_labels = torch.zeros([len(val_ls), N]).to(device)

    # for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
    # max_tt_len = max([tt for tt in tt_ls])
    for b in range(len(tt_ls)):
        tt, vals, mask = tt_ls[b], val_ls[b], mask_ls[b]#, labels_ls[b]
        if test_mask_ls is not None:
            test_mask = test_mask_ls[b]
        vals[mask==0] = 0
        currlen = tt.size(0)
        if not right_align:
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)
            if test_mask_ls is not None:
                enc_combined_test_mask[b, :currlen] = test_mask.to(device)
            # if classify:
            # if activity:
            #     combined_labels[b, :currlen] = labels.to(device)
            # else:
            #     combined_labels[b] = labels.to(device)
        else:
            enc_combined_tt[b, maxlen - currlen:] = tt.to(device)
            enc_combined_vals[b, maxlen - currlen:] = vals.to(device)
            enc_combined_mask[b, maxlen - currlen:] = mask.to(device)
            if test_mask_ls is not None:
                enc_combined_test_mask[b, maxlen - currlen:] = test_mask.to(device)
            # if classify:
            # if activity:
            #     combined_labels[b, maxlen - currlen:] = labels.to(device)
            # else:
            #     combined_labels[b] = labels.to(device)

    if not activity:
        if data_min is not None and data_max is not None:
            enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    # if classify:
    return combined_data, enc_combined_test_mask


def get_data_min_max(values, masks):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]#.to(device)

    for idx in range(len(values)):
        
        vals = values[idx]
        mask = masks[idx]
        n_features = vals.shape[-1]

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:,i][mask[:,i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

class mTan_dataset(torch.utils.data.Dataset):
    def __init__(self, visit_tensor_ls, mask_ls, time_step_ls, label_tensor_ls, data_min, data_max, all_test_mask_ls):
        
        self.visit_tensor_ls = visit_tensor_ls
        self.mask_ls = mask_ls
        self.time_step_ls = time_step_ls
        
        self.label_tensor_ls = label_tensor_ls
        self.data_min = data_min
        self.data_max = data_max
        self.all_test_mask_ls = all_test_mask_ls


        # self.data_tensor, self.label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
        # assert len(self.label_tensor_ls) == len(self.person_info_ls)
        # assert len(self.person_info_ls) == len(self.time_step_ls)
        assert len(self.visit_tensor_ls) == len(self.label_tensor_ls)

    def __len__(self):
        return len(self.time_step_ls)

    
    def __getitem__(self, idx):
        # return self.data_tensor[idx], self.label_tensor[idx]
        if self.all_test_mask_ls is not None:
            return self.visit_tensor_ls[idx], self.mask_ls[idx], self.time_step_ls[idx], self.label_tensor_ls[idx], self.data_min, self.data_max, self.all_test_mask_ls[idx]
        else:
            return self.visit_tensor_ls[idx], self.mask_ls[idx], self.time_step_ls[idx], self.label_tensor_ls[idx], self.data_min, self.data_max, None


    @staticmethod
    def collate_fn(data):
        time_step_ls = [item[2] for item in data]
        visit_tensor_ls = [item[0] for item in data]
        mask_ls = [item[1] for item in data]
        label_tensor_ls = [item[3] for item in data]
        # person_info_ls = [item[3].view(-1) for item in data]
        data_min = [item[4] for item in data][0]
        data_max = [item[5] for item in data][0]
        test_mask_ls = None
        if data[0][6] is not None:
            test_mask_ls = [item[6] for item in data]
            
        batched_data_tensor, batched_label_tensor, combined_test_mask = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max, test_mask_ls = test_mask_ls)
        # batched_person_=[]
        # batched_person_info = torch.stack(person_info_ls)
        # batched_data_tensor = torch.stack([item[0] for item in data])
        # batched_label_tensor = torch.stack([item[1] for item in data])
        return batched_data_tensor, batched_label_tensor, combined_test_mask