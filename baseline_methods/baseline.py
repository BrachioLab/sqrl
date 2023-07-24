import torch
import torch.nn.functional as F



def entropy_classification_loss(p_avg):
    tta_loss = - (p_avg * torch.log(p_avg)).sum(dim=1)

    return tta_loss.mean()

def diff_iou_scores(bbox1, bbox2):
    X = (bbox1[:,0] + bbox1[:,2])*(bbox1[:,1] + bbox1[:,3])
    X_bar = (bbox2[:,0] + bbox2[:,2])*(bbox2[:,1] + bbox2[:,3])
    I_h = torch.min(bbox1.unsqueeze(0)[:,:, 0], bbox2.unsqueeze(1)[:,:, 0]) + torch.min(bbox1.unsqueeze(0)[:,:, 2], bbox2.unsqueeze(1)[:,:, 2])
    I_w = torch.min(bbox1.unsqueeze(0)[:,:, 1], bbox2.unsqueeze(1)[:,:, 1]) + torch.min(bbox1.unsqueeze(0)[:,:,3], bbox2.unsqueeze(1)[:,:, 3])
    I = I_h*I_w
    U = X.unsqueeze(0) + X_bar.unsqueeze(1) - I
    IOU = I/U
    return IOU
    # loss = -torch.mean(torch.log(IOU))
    # return loss

def l2_consistency_loss(loss_ls):
    full_loss = 0
    pair_count= 0
    for idx1 in range(len(loss_ls)):
        for idx2 in range(len(loss_ls)):
            if idx1 != idx2:
                bbox1 = loss_ls[idx1]
                bbox2 = loss_ls[idx2]
                bbox1[:,2] = bbox1[:,0] + bbox1[:,2]
                bbox1[:,3] = bbox1[:,1] + bbox1[:,3]
                bbox2[:,2] = bbox2[:,0] + bbox2[:,2]
                bbox2[:,3] = bbox2[:,1] + bbox2[:,3]

                # iou_scores = bbox_overlaps(bbox1, bbox2)
                iou_scores = diff_iou_scores(bbox1, bbox2)
                full_loss += -torch.min(torch.max(iou_scores, dim=1)[0]) - torch.min(torch.max(iou_scores, dim=0)[0])

                # full_loss = ((loss_ls[idx1] - loss_ls[idx2])**2)
                pair_count += 1
    return full_loss/pair_count


# copied from https://github.com/locuslab/tta_conjugate/blob/main/imagenet_tta_test.py#L161

def robust_pl(outputs, input_probs = False):
    if not input_probs:
        p = F.softmax(outputs, dim=1)
    else:
        p = outputs
    y_pl = outputs.max(1)[1]
    Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
    tta_loss = (1- (Yg**0.8))/0.8

    return tta_loss.mean()

def conjugate_pl(outputs, num_classes=1000, eps=8.0, input_probs=False):
    if not input_probs:
        softmax_prob = F.softmax(outputs, dim=1)
    else:
        softmax_prob = outputs
    smax_inp = softmax_prob 

    eye = torch.eye(num_classes).to(outputs.device)
    eye = eye.reshape((1, num_classes, num_classes))
    eye = eye.repeat(outputs.shape[0], 1, 1)
    t2 = eps * torch.diag_embed(smax_inp)
    smax_inp = torch.unsqueeze(smax_inp, 2)
    t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
    matrix = eye + t2 - t3
    y_star = torch.solve(smax_inp, matrix)[0]
    y_star = torch.squeeze(y_star)

    pseudo_prob = y_star
    tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)

    return tta_loss.mean()



def memo_loss(logits_aug1, logits_aug2, logits_aug3, T=1.0):
    if len(logits_aug1.shape) >= 2 and logits_aug1.shape[-1] > 1:
        p_aug1, p_aug2, p_aug3 = F.softmax(logits_aug1/T, dim=1), F.softmax(logits_aug2/T, dim=1), F.softmax(logits_aug3/T, dim=1)
    else:
        logits_aug1 = logits_aug1.view(-1)
        logits_aug2 = logits_aug2.view(-1)
        logits_aug3 = logits_aug3.view(-1)
        p_aug1, p_aug2, p_aug3 = F.sigmoid(logits_aug1/T), F.sigmoid(logits_aug2/T), F.sigmoid(logits_aug3/T)

        p_aug1 = torch.stack([1-p_aug1, p_aug1], dim=1)
        p_aug2 = torch.stack([1-p_aug2, p_aug2], dim=1)
        p_aug3 = torch.stack([1-p_aug3, p_aug3], dim=1)
    p_avg = (p_aug1 + p_aug2 + p_aug3) / 3
    tta_loss = - (p_avg * torch.log(p_avg)).sum(dim=1)
    tta_loss = tta_loss.mean()
    return tta_loss
