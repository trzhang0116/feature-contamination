import torch
from .model import FullModel


def get_model(args, from_scratch=False, include_top=False, device=torch.device('cuda')):
    return FullModel(args,
                     args.model_name,
                     1000,
                     args.n_layers,
                     args.reversible,
                     args.residual,
                     args.n_nonlinear_transform_blocks,
                     args.feature_cache_dir,
                     from_scratch=from_scratch,
                     include_top=include_top,
                     device=device
                     )