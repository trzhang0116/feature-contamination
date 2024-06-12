import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image

import templates
import datasets
from .clip.clip import load, tokenize, get_clip_transform
from .mlp import ClassificationHead

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def copy_parameter(source, target):
    if isinstance(source, nn.Module):
        target.load_state_dict(source.state_dict())
    elif isinstance(source, nn.Parameter):
        target.data.copy_(source.data)
    else:
        raise ValueError(f'Unexpected type: {type(source)}')


def initialize_attention_layers(source_transformer, target_transformer):
    # Initialize the attention layers in the target transformer using the weights of source transformer
    n_blocks = len(source_transformer.resblocks)
    for i in range(n_blocks):
        source_block = source_transformer.resblocks[i]
        target_block = target_transformer.resblocks[i]
        source_layer = source_block.attn
        target_layer = target_block.attn
        copy_parameter(source_layer, target_layer)
    print(f'Initialized attention layer in block(s) {np.arange(n_blocks)}')


def get_zeroshot_classifier(args, input_dim, output_dim, device):
    assert args.template is not None
    assert args.train_dataset is not None

    clip_model = load(args.model_name.split('_')[1], device=device)[0]
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        None,
        location=args.data_dir,
        batch_size=args.batch_size,
        classnames=args.classnames
    )
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(input_dim, output_dim, weights=zeroshot_weights, normalize=True)
    torch.save(classification_head.state_dict(), f'{args.result_dir}/models/zeroshot.pt')

    return classification_head


def get_transform(args, input_resolution, aug=False):
    if not 'clip' in args.model_name.lower():
        if aug:
            return transforms.Compose([transforms.RandomResizedCrop(input_resolution, scale=(0.9, 1.0), interpolation=BICUBIC),  
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                       ])
        else:
            return transforms.Compose([transforms.Resize(input_resolution, interpolation=BICUBIC),
                                       transforms.CenterCrop(input_resolution),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                       ])
    else:
        return get_clip_transform(input_resolution, is_train=aug)


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return input.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)

    def forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
