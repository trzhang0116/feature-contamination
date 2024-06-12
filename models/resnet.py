import torch.nn as nn
import torchvision.models as models


class ResNetFeaturizer(nn.Module):

    def __init__(self,
                 model_name,
                 version=None,
                 cache_dir=None,
                 include_top=False,
                 from_scratch=False):
        
        super().__init__()

        if from_scratch:
            self.model = getattr(models, model_name.lower())(weights=None)
        else:
            if version.lower() == 'v1':
                self.model = getattr(models, model_name.lower())(weights='IMAGENET1K_V1')  # V1
            else:
                self.model = getattr(models, model_name.lower())(weights='DEFAULT')  # V2
            self._freeze_bn()

        self.input_resolution = 224
        self.output_dim = self.model.fc.in_features
        self.cache_dir = cache_dir
        self.include_top = include_top
        self.from_scratch = from_scratch

        if self.include_top:
            self.linear = self.model.fc

        self.model.fc = nn.Identity()
        
        self._set_requires_grad(from_scratch)

    def forward(self, x):
        x = self.model(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        if not self.from_scratch:
            self._freeze_bn()
    
    def eval(self):
        self.train(False)

    def _freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def _set_requires_grad(self, status):
        for name, params in self.model.named_parameters():
            if not 'linear' in name:
                params.requires_grad = status
    