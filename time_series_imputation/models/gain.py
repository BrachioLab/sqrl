import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)

class Gain(nn.Module):
    def __init__(self, Dim, H_Dim1, H_Dim2):
        super(Gain, self).__init__()
        self.theta_D = nn.Sequential(nn.Linear(Dim*2, H_Dim1), nn.ReLU(), nn.Linear(H_Dim1, H_Dim2), nn.ReLU(), nn.Linear(H_Dim2, Dim))
        self.theta_G = nn.Sequential(nn.Linear(Dim*2, H_Dim1), nn.ReLU(), nn.Linear(H_Dim1, H_Dim2), nn.ReLU(), nn.Linear(H_Dim2, Dim))
        self.init_weight()
    def init_weight(self):
        for layer in list(self.theta_D.children()):
            if hasattr(layer, "weight"):
                torch.nn.init.xavier_uniform(layer.weight)
        for layer in list(self.theta_G.children()):
            if hasattr(layer, "weight"):
                torch.nn.init.xavier_uniform(layer.weight)
    #%% 1. Generator
    def generator(self, new_x,m):
        inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
        G_prob = torch.sigmoid(self.theta_D(inputs))
        # G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        # G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
        # torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
        
        return G_prob
    def discriminator(self, new_x, h):
        inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
        D_prob = torch.sigmoid(self.theta_D(inputs))
        # D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
        # D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        # D_logit = torch.matmul(D_h2, D_W3) + D_b3
        # D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        
        return D_prob

    def discriminator_loss(self, M, New_X, H):
        # Generator
        G_sample = self.generator(New_X,M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        #%% Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def generator_loss(self, X, M, New_X, H, alpha = 10):
        #%% Structure
        # Generator
        G_sample = self.generator(New_X,M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        #%% Loss
        G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

        G_loss = G_loss1 + alpha * MSE_train_loss 

        #%% MSE Performance metric
        MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return G_loss, MSE_train_loss, MSE_test_loss
        
    def test_loss(self, X, M, New_X):
        #%% Structure
        # Generator
        G_sample = self.generator(New_X,M)

        #%% MSE Performance metric
        MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return MSE_test_loss, G_sample