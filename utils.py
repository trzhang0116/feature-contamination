import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import collections


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr_with_warmup(optimizer, base_lrs, warmup_length, steps, warm_restarts=False, restart_steps=0):
    if isinstance(optimizer, list):
        return [cosine_lr_with_warmup(optim, base_lrs, warmup_length, steps) for optim in optimizer]
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                if not warm_restarts:
                    e = step - warmup_length
                    es = steps - warmup_length
                else:
                    if step > restart_steps:
                        e = step % restart_steps
                        es = restart_steps
                    else:
                        e = step - warmup_length
                        es = restart_steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            param_group['lr'] = lr
    return _lr_adjuster


def remove_prefix_in_checkpoints(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unnormalize(imgs, model_name):
    # imgs: [N, H, W, C]
    # for each channel in C:
    #     output[..., channel] = input[..., channel] * std[channel] + mean[channel]
    if 'clip' in model_name.lower():
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    for i in range(3):
        imgs[..., i] = imgs[..., i] * std[i] + mean[i]
    
    imgs = np.clip(imgs, a_min=0, a_max=1)
    
    return imgs


class KDLoss(nn.Module):
    """
    Distilling the knowledge in a neural network, https://arxiv.org/abs/1503.02531
    """
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T
    
    def forward(self, logits_student, logits_teacher):
        return F.kl_div(F.log_softmax(logits_student/self.T, dim=1), F.softmax(logits_teacher/self.T, dim=1), reduction='batchmean') * self.T**2


class HintLoss(nn.Module):
    """
    FitNets: Hints for thin deep nets, https://arxiv.org/abs/1412.6550
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z_student, z_teacher):
        # z_teacher: [N, D]
        # z_student: [N, D]
        return F.mse_loss(z_student, z_teacher)
