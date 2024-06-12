import torch
import torch.nn as nn
import torch.linalg as LA
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import set_seed


# general settings
loss_name    = 'logistic'    # 'Logistic', 'Hinge', or 'MSE'
opt          = 'adamw'       # 'SGD', 'Adam', or 'AdamW'
opt_momentum = 0.9           # momentum for SGD

# data parameters
n         = 1000     # number of data points
core_dim  = 32       # dimension of core features
bg_dim    = 32       # dimension of background features
input_dim = 256      # input dimension
nonlinear = True     # non-linear relationships between core features and labels

# network parameters
hidden_dim        = 256       # hidden layer dimension
mlp_n_layers      = 2         # number of layers in MLP
mlp_act           = 'relu'    # 'ReLU', 'GELU', 'LeakyReLU', 'Sigmoid', or 'Tanh'
mlp_freeze_output = False     # whether to freeze the output layer of MLP    

# training parameters
gid           = 0        # GPU ID
train_iters   = 20000    # training iterations
learning_rate = 1e-3
weight_decay  = 1e-3

# plot settings
title_fontsize  = 24
label_fontsize  = 20
tick_fontsize   = 16
legend_fontsize = 20
linewidth       = 3

# other settings
seed             = 2024
print_freq       = 1000
plot_corr        = True
plot_activation  = False
save_dir         = './results/toy/'


class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features=1, bias=True):
        super().__init__()
        
        assert mlp_n_layers >= 2
        
        self.fcs = nn.ModuleList([nn.Linear(in_features, hidden_features, bias=bias)] \
            + [nn.Linear(hidden_features, hidden_features, bias=bias) for _ in range(mlp_n_layers-2)] \
            + [nn.Linear(hidden_features, out_features, bias=bias)])
        
        if mlp_freeze_output:
            self.fcs[-1].weight.data[:, :hidden_dim // 2] = 1. / hidden_dim * torch.ones(out_features, hidden_dim//2)
            self.fcs[-1].weight.data[:, hidden_dim // 2:] = -1. / hidden_dim * torch.ones(out_features, hidden_dim-hidden_dim//2)
            for param in self.fcs[-1].parameters():
                param.requires_grad = False        
        
        if mlp_act.lower() == 'relu':
            self.act = nn.ReLU()
        elif mlp_act.lower() == 'gelu':
            self.act = nn.GELU()
        elif mlp_act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)
        elif mlp_act.lower() == 'sigmoid':
            self.act = nn.Sigmoid()
        elif mlp_act.lower() == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.fcs[0](x)
        for i in range(1, mlp_n_layers):
            x = self.act(x)
            x = self.fcs[i](x)
        return x


def generate_data(input_transform):
    N_pos = n // 2
    N_neg = n - N_pos
    y = torch.cat([torch.ones(N_pos), -torch.ones(N_neg)], dim=0)
    
    if nonlinear: 
        Xc_pos = MultivariateNormal(loc=torch.zeros(core_dim), covariance_matrix=torch.eye(core_dim)).sample((N_pos,))
        Xc_pos = Xc_pos / Xc_pos.norm(dim=-1, keepdim=True)
        Xc_neg = MultivariateNormal(loc=torch.zeros(core_dim), covariance_matrix=torch.eye(core_dim)).sample((N_neg,))
        Xc_neg = 2 * Xc_neg / Xc_neg.norm(dim=-1, keepdim=True)
    else:
        Xc_pos = torch.rand(N_pos, core_dim)
        Xc_neg = -torch.rand(N_neg, core_dim)
    
    Xc = torch.cat([Xc_pos, Xc_neg], dim=0)
    
    Xs_train = torch.rand(n, bg_dim)
    Xs_test = -torch.rand(n, bg_dim)
    train_x = torch.cat([Xc, Xs_train], dim=-1)
    test_x = torch.cat([Xc, Xs_test], dim=-1)
    
    train_x = train_x @ input_transform
    test_x = test_x @ input_transform
    
    return train_x, test_x, y


def random_orthogonal_transform(input_size, output_size):
    random_matrix = torch.randn(input_size, output_size)
    svd = LA.svd(random_matrix, full_matrices=False)
    if input_size >= output_size:
        return svd[0]  # return U
    return svd[2]  # return VH


class LogisticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return torch.mean(torch.log(1 + torch.exp(-y * y_pred)))


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y):
        return torch.mean(torch.clamp(1 - y * y_pred, min=0))    


def get_loss():
    if loss_name.lower() == 'logistic':
        return LogisticLoss()
    elif loss_name.lower() == 'hinge':
        return HingeLoss()
    elif loss_name.lower() == 'mse':
        return nn.MSELoss()


def get_optimizer(model):
    if opt.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=opt_momentum, weight_decay=weight_decay)
    elif opt.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    

def main():
    set_seed(seed)
    device = torch.device(f'cuda:{gid}' if torch.cuda.is_available() else 'cpu')
    
    model = MLP(in_features=input_dim, hidden_features=hidden_dim).train().to(device)
    optimizer = get_optimizer(model)
    loss_func = get_loss()
    
    input_transform = random_orthogonal_transform(core_dim + bg_dim, input_dim)
    train_x, test_x, y = generate_data(input_transform)
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    y = y.to(device)
    
    # Training
    train_losses = []
    eval_losses = []
    corr_core = []
    corr_spu = []
    
    for i in tqdm(range(train_iters)):
        pred = model(train_x)
        loss = loss_func(pred.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_freq == 0 or i == train_iters - 1:
            eval_loss = evaluate(model, test_x, y, loss_func)
            train_losses.append(loss.item())
            eval_losses.append(eval_loss)
            print(f'Training iteration {i}: train loss = {loss.item():.4f}, eval loss = {eval_loss:.4f}')
            if plot_corr:
                weight = model.fcs[0].weight.cpu().detach()
                weight = weight @ input_transform.T
                corr_core.append(LA.norm(weight[:, :core_dim], dim=-1).mean().item())
                corr_spu.append(LA.norm(weight[:, core_dim:], dim=-1).mean().item())
    
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    corr_core = np.array(corr_core)
    corr_spu = np.array(corr_spu)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save stats
    np.savez(os.path.join(save_dir, f'{loss_name}_{opt}_{mlp_act}.npz'),
             train_loss=train_losses, eval_loss=eval_losses,
             corr_core=corr_core, corr_spu=corr_spu)
    
    # plot
    plot_loss_and_weights(train_losses, eval_losses, corr_core, corr_spu)


def plot_multiple_activations():
    # load data
    data = {}
    for act in ['ReLU', 'GELU', 'Sigmoid', 'Tanh']:
        data[act] = np.load(os.path.join(save_dir, f'{loss_name}_{opt}_{act}.npz'))
    
    # plot
    mpl.rcParams['axes.linewidth'] = 2.
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    for act in ['ReLU', 'GELU', 'Sigmoid', 'Tanh']:
        corr_spu = data[act]['corr_spu']
        ax.plot(print_freq * np.arange(len(corr_spu)), corr_spu, label=act, linewidth=linewidth)
    
    ax.set_xlabel('Iterations', fontdict={'fontsize': label_fontsize})
    ax.set_ylabel('Avg. BG. Feature Corr.', fontdict={'fontsize': label_fontsize})
    ax.legend(fontsize=legend_fontsize, loc='lower right')
    ax.grid()
    ax.xaxis.set_tick_params(labelsize=tick_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{loss_name}_{opt}_activations.pdf'))
    plt.show()


def evaluate(model, test_x, y, criterion):
    model.eval()
    with torch.no_grad():
        y_pred = model(test_x)
        if isinstance(y_pred, tuple):
            y_pred, _ = y_pred
        loss = criterion(y_pred.squeeze(), y)
    model.train()
    return loss.item()


def plot_loss_and_weights(train_losses, eval_losses, corr_core, corr_spu):
    mpl.rcParams['axes.linewidth'] = 2.
    if len(corr_core) > 0:
        f, axs = plt.subplots(1, 2, figsize=(10, 4))
    else:
        f, axs = plt.subplots(1, figsize=(5, 4))
    
    # plot losses
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    axs[0].plot(print_freq * np.arange(len(train_losses)), train_losses, label='ID loss', linewidth=linewidth)
    axs[0].plot(print_freq * np.arange(len(eval_losses)), eval_losses, label='OOD loss', linewidth=linewidth)
    axs[0].set_xlabel('Iterations', fontdict={'fontsize': label_fontsize})
    axs[0].set_ylabel('Loss', fontdict={'fontsize': label_fontsize})
    axs[0].legend(fontsize=legend_fontsize, loc='lower right')
    axs[0].grid()
    axs[0].xaxis.set_tick_params(labelsize=tick_fontsize)
    axs[0].yaxis.set_tick_params(labelsize=tick_fontsize)

    # plot weights
    if len(corr_core) > 0:
        axs[-1].plot(print_freq * np.arange(len(corr_core)), corr_core, label='Core', linewidth=linewidth)
        axs[-1].plot(print_freq * np.arange(len(corr_spu)), corr_spu, label='Background', linewidth=linewidth)
        axs[-1].set_xlabel('Iterations', fontdict={'fontsize': label_fontsize})
        axs[-1].set_ylabel('Avg. Feature Corr.', fontdict={'fontsize': label_fontsize})
        axs[-1].legend(fontsize=legend_fontsize, loc='lower right')
        axs[-1].grid()
        axs[-1].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[-1].yaxis.set_tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    aux_str = 'nonlinear' if nonlinear else 'linear'
    
    plt.savefig(os.path.join(save_dir, f'{loss_name}_{opt}_{mlp_act}_{aux_str}.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
    # plot_multiple_activations()
   