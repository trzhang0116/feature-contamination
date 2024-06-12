import torch
import torch.nn as nn

from .clip.clip import CLIPFeaturizer
from .resnet import ResNetFeaturizer
from .mlp import MLP, ClassificationHead
from .utils import get_zeroshot_classifier


class FullModel(nn.Module):

    def __init__(
        self,
        args,
        model_name,
        n_classes,
        n_featurizer_layers,
        reversible_featurizer,
        residual,
        n_nonlinear_transform_blocks,
        feature_cache_dir,
        from_scratch=False,
        include_top=False,
        device=torch.device('cuda')):

        super().__init__()

        self.include_top = include_top
        
        if 'clip' in model_name.lower():
            # example model name: CLIP_RN50
            model_name = model_name.split('_')[1]
            if args.attn_distill:
                attn_weight_inds = [int(i) if i != 'last' else -1 for i in args.attn_distill_blocks]
                print(f'Using attention map distillation for block(s) {attn_weight_inds}.')
            else:
                attn_weight_inds = None
            if args.layer_distill:
                hidden_state_inds = [int(i) if i != 'last' else -1 for i in args.layer_distill_blocks]
                print(f'Using layer-wise distillation for block(s) {hidden_state_inds}.')
            else:
                hidden_state_inds = None
            
            self.pre_featurizer = CLIPFeaturizer(
                model_name, feature_cache_dir, device=device, from_scratch=from_scratch,
                attn_weight_inds=attn_weight_inds, hidden_state_inds=hidden_state_inds).to(device)
        
        elif 'erm' in model_name.lower():
            # example model name: ERM_ResNet50/ERM_ResNet50_V2
            splits = model_name.split('_')
            model_name = splits[1]
            version = splits[2] if len(splits) == 3 else None
            self.pre_featurizer = ResNetFeaturizer(
                model_name, version, feature_cache_dir, include_top, from_scratch=from_scratch).to(device)

        self.input_resolution = self.pre_featurizer.input_resolution
        self.feature_dim = self.pre_featurizer.output_dim
        
        # random non-linear transformations of features
        if n_nonlinear_transform_blocks > 0:
            nonlinear_transform = MLP(
                self.feature_dim, self.feature_dim, n_nonlinear_transform_blocks,
                reversible=True, residual=False, requires_grad=False)
            self.pre_featurizer.register_transform(nonlinear_transform)
        
        # additional feature extraction layers after the encoder (pre_featurizer)
        if n_featurizer_layers == 0:
            self.featurizer = nn.Identity()
        else:
            self.featurizer = MLP(self.feature_dim, self.feature_dim, n_featurizer_layers,
                                  reversible_featurizer, residual)
        
        if include_top:
            if hasattr(self.pre_featurizer, 'linear'):
                self.classification_head = self.pre_featurizer.linear
            else:
                if isinstance(self.pre_featurizer, CLIPFeaturizer) and args.zeroshot_init:
                    print('Using zero-shot classifier as classification head initialization.')
                    self.classification_head = get_zeroshot_classifier(args, self.feature_dim, n_classes, device)
                else:
                    self.classification_head = ClassificationHead(self.feature_dim, n_classes)

        self.full_forward = False

    def forward(self, x, feat=False):
        if self.full_forward:
            feats = self.pre_featurizer(x)
        else:
            feats = x
        
        feats = self.featurizer(feats)

        if not self.include_top:
            return feats
        
        x = self.classification_head(feats)
        
        if feat:
            return x, feats
        return x

    def train(self, mode=True):
        self.featurizer.train(mode)
        if self.include_top:
            self.classification_head.train(mode)

    def eval(self):
        self.train(False)
    
    def set_full_forward(self, mode=True):
        self.full_forward = mode
