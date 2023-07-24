#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2, has_static=False, static_input_dim=0):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, N))

        if has_static:
            self.static_feat = nn.Sequential(
                nn.Linear(static_input_dim, 20),
                nn.ReLU(),
                nn.Linear(20, nhidden))
            self.classifier = nn.Sequential(
                nn.Linear(2*nhidden, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, N))
        
       
    def forward(self, z, static_x=None):
        _, out = self.gru_rnn(z)
        # if static_x is not None:
        if static_x is not None:
            static_feat = self.static_feat(static_x)
            return self.classifier(torch.cat([out.squeeze(0), static_feat], dim = -1))
        else:
            return self.classifier(out.squeeze(0))

    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)

class mtan_model_full(nn.Module):
    def __init__(self, rec, dec, model_config, device):
        super(mtan_model_full, self).__init__()
        self.dec = dec
        self.rec = rec
        # self.classifier = classifier
        self.model_config = model_config
        self.device = device
    
    def log_normal_pdf(self, x, mean, logvar, mask):
        const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
        const = torch.log(const)
        return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask

    def normal_kl(self, mu1, lv1, mu2, lv2):
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        return kl
    
    def compute_losses(self, observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, device):
        # observed_data, observed_mask \
        #     = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

        noise_std = self.model_config["std"]  # default 0.1
        noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = self.log_normal_pdf(observed_data, pred_x, noise_logvar,
                            observed_mask).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
        analytic_kl = self.normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1).sum(-1)
        if self.model_config["norm"]:
            logpx /= observed_mask.sum(-1).sum(-1)
            analytic_kl /= observed_mask.sum(-1).sum(-1)
        del noise_std_, pz0_mean, pz0_logvar, noise_logvar
        return logpx, analytic_kl

        
    def forward(self, subsampled_data, subsampled_tp, subsampled_mask, kl_coef=0):
        out = self.rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
        qz0_mean = out[:, :, :self.model_config["latent_dim"]]
        qz0_logvar = out[:, :, self.model_config["latent_dim"]:]
        # epsilon = torch.randn(qz0_mean.size()).to(device)
        epsilon = torch.randn(
            self.model_config["k_iwae"], qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
        ).to(self.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_x = self.dec(
            z0,
            subsampled_tp[None, :, :].repeat(self.model_config["k_iwae"], 1, 1).view(-1, subsampled_tp.shape[1])
        )
        # nsample, batch, seqlen, dim
        pred_x = pred_x.view(self.model_config["k_iwae"], subsampled_data.shape[0], pred_x.shape[1], pred_x.shape[2])
        # compute loss
        logpx, analytic_kl = self.compute_losses(
            subsampled_data, subsampled_mask, qz0_mean, qz0_logvar, pred_x, self.device)
        loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(self.model_config["k_iwae"]))
        # del epsilon, logpx, analytic_kl, z0, qz0_mean, qz0_logvar, out
        return loss#, pred_x.detach().cpu()
    
    
    def evaluate(self, subsampled_data, subsampled_tp, subsampled_mask, gt_mask, num_sample =1):
        subsampled_mask = gt_mask
        out = self.rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
        qz0_mean, qz0_logvar = (
            out[:, :, : self.model_config["latent_dim"]],
            out[:, :, self.model_config["latent_dim"]:],
        )
        epsilon = torch.randn(
            num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
        ).to(self.device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        batch, seqlen = subsampled_tp.size()
        time_steps = (
            subsampled_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
        )
        pred_x = self.dec(z0, time_steps)
        pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
        pred_x = pred_x.mean(0)
        return pred_x
    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        
   
    
class enc_mtan_classif(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2))
        self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))




class enc_mtan_classif_activity(nn.Module):
 
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif_activity, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        self.classifier = nn.Linear(nhidden, 11)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        batch = x.size(0)
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
        out = self.att(key, key, x, mask)
        out, _ = self.gru(out)
        out = self.classifier(out)
        return out
    
    
    
class enc_interp(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(enc_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.cross = nn.Linear(2*input_dim, 2*input_dim)
        self.bandwidth = nn.Linear(1, 2*input_dim, bias=False)
        self.gru_rnn = nn.GRU(2*input_dim, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))

    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
        
    def forward(self, x, time_steps):
        #time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        out, _ = self.attention(self.query.unsqueeze(0), time_steps, x, mask)
        out = self.cross(out)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_interp(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(dec_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.bandwidth = nn.Linear(1, 2*nhidden, bias=False)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
        
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        out, _ = self.attention(time_steps, self.query.unsqueeze(0), out)
        out = self.z0_to_obs(out)
        return out        

    
class enc_rnn3(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, use_classif=False, learn_emb=False, device='cuda'):
        super(enc_rnn3, self).__init__()
        self.use_classif = use_classif 
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.cross = nn.Linear(2*input_dim, nhidden)
        if use_classif:
            self.gru_rnn = nn.GRU(nhidden, nhidden, batch_first=True)
        else:
            self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time) for _ in range(2)])
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        if use_classif:
            self.classifier = nn.Sequential(
                nn.Linear(nhidden, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 2))
        else:
            self.hiddens_to_z0 = nn.Sequential(
                nn.Linear(2*nhidden, 50),
                nn.ReLU(),
                nn.Linear(50, latent_dim * 2))
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device) 
        out, _ = self.attention(query, key, x, mask)
        out = self.cross(out)
        if not self.use_classif:
            out, _ = self.gru_rnn(out)
            out = self.hiddens_to_z0(out)
        else:
            _, h = self.gru_rnn(out)
            out = self.classifier(h.squeeze(0))
        return out
    
    
class dec_rnn3(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16,learn_emb=False, device='cuda'):
        super(dec_rnn3, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                     nn.Linear(embed_time, embed_time),
                                     nn.Linear(2*nhidden, 2*nhidden)])
      
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out, _ = self.attention(query, key, out)
        out = self.z0_to_obs(out)
        return out        
    
