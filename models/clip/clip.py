# Code modified from https://github.com/openai/CLIP

import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_to_rgb(image):
    return image.convert('RGB')


def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=BICUBIC),
            RandomHorizontalFlip(),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


def get_clip_transform(n_px: int, is_train: bool = False):
    return _transform(n_px, is_train)


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
        else:
            reset_parameters = getattr(m, "_reset_parameters", None)  # multi-head attention
            if callable(reset_parameters):
                m._reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

    # reset the Attention Pooling layer of modified ResNet
    if hasattr(model.visual, 'attnpool'):
        positional_embedding = model.visual.attnpool.positional_embedding
        spacial_dim, embed_dim = positional_embedding.data.size()
        positional_embedding.data = torch.randn(spacial_dim, embed_dim) / embed_dim ** 0.5
    
    # reset extra parameters of vision transformer
    if hasattr(model.visual, 'class_embedding'):
        class_embedding = model.visual.class_embedding
        width = class_embedding.size(0)
        scale = width ** -0.5
        class_embedding.data = scale * torch.randn(width)

        positional_embedding = model.visual.positional_embedding
        h, w = positional_embedding.data.size()
        positional_embedding.data = scale * torch.randn(h, w)

        proj = model.visual.proj
        h, w = proj.data.size()
        proj.data = scale * torch.randn(h, w)
    

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False, is_train=False, pretrained=True,
         attn_weight_inds=None, hidden_state_inds=None):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model(state_dict or model.state_dict()).to(device)
        except KeyError:
            sd = {k[7:]: v for k,v in state_dict["state_dict"].items()}
            model = build_model(sd).to(device)

        if str(device) == "cpu":
            model.float()
        
        if not pretrained:
            reset_all_weights(model)

        if is_train:
            model.train()
        
        if attn_weight_inds is not None:
            assert 'vit' in name.lower()
            model.visual.aux_outputs = True
            for i in attn_weight_inds:
                model.visual.transformer.resblocks[i].need_attn_weights = True
        
        if hidden_state_inds is not None:
            assert 'vit' in name.lower()
            model.visual.aux_outputs = True
            for i in hidden_state_inds:
                model.visual.transformer.resblocks[i].need_hidden_states = True
        
        return model, \
               _transform(model.visual.input_resolution, is_train=True), \
               _transform(model.visual.input_resolution, is_train=False)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()
    
    if not pretrained:
        reset_all_weights(model)

    if is_train:
        model.train()
    
    if attn_weight_inds is not None:
        assert 'vit' in name.lower()
        model.visual.aux_outputs = True
        for i in attn_weight_inds:
            model.visual.transformer.resblocks[i].need_attn_weights = True
    
    if hidden_state_inds is not None:
        assert 'vit' in name.lower()
        model.visual.aux_outputs = True
        for i in hidden_state_inds:
            model.visual.transformer.resblocks[i].need_hidden_states = True

    return model, \
           _transform(model.input_resolution.item(), is_train=True), \
           _transform(model.input_resolution.item(), is_train=False)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class CLIPFeaturizer(nn.Module):
    
    def __init__(self,
                 model_name,
                 cache_dir,
                 transform=None,
                 device=torch.device('cuda'),
                 from_scratch=False,
                 attn_weight_inds=None,
                 hidden_state_inds=None
                 ):
        
        super().__init__()

        if from_scratch:
            self.model = load(model_name, device=device, jit=False, is_train=True, pretrained=False,
                              attn_weight_inds=attn_weight_inds, hidden_state_inds=hidden_state_inds)[0]
        else:
            self.model = load(model_name, device=device, jit=False, attn_weight_inds=attn_weight_inds,
                              hidden_state_inds=hidden_state_inds)[0]
        
        self.input_resolution = self.model.visual.input_resolution

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
        if hasattr(self.model, 'token_embedding'):
            delattr(self.model, 'token_embedding')
        if hasattr(self.model, 'positional_embedding'):
            delattr(self.model, 'positional_embedding')
        if hasattr(self.model, 'ln_final'):
            delattr(self.model, 'ln_final')
        if hasattr(self.model, 'text_projection'):
            delattr(self.model, 'text_projection')
        if hasattr(self.model, 'logit_scale'):
            delattr(self.model, 'logit_scale')
        
        self.model = self.model.float()
        self.output_dim = self.model.visual.output_dim
        self.from_scratch = from_scratch

        self._set_requires_grad(from_scratch)

        self.transform = transform
        self.cache_dir = cache_dir
        self.aux_outputs = False if attn_weight_inds is None and hidden_state_inds is None else True
        self.norm_oracle = False

    def _set_requires_grad(self, status):
        for params in self.model.parameters():
            params.requires_grad = status
    
    def _freeze_bn(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    def register_transform(self, transform):
        self.transform = transform
    
    def train(self, mode):
        super().train(mode)
        if not self.from_scratch or self.norm_oracle:
            self._freeze_bn()
    
    def load_bn_stats(self, model):
        self.norm_oracle = True
        for source_module, target_module in zip(self.model.modules(), model.modules()):
            if isinstance(target_module, nn.BatchNorm2d):
                assert isinstance(source_module, nn.BatchNorm2d)
                target_module.running_mean = source_module.running_mean
                target_module.running_var = source_module.running_var
                target_module.eval()
    
    def eval(self):
        self.train(False)
    
    def forward(self, x):
        if self.aux_outputs:
            y, aux_outputs = self.model.encode_image(x)
            if self.transform is not None:
                y = self.transform(y)
            return y, aux_outputs
        
        y = self.model.encode_image(x)
        if self.transform is not None:
            y = self.transform(y)
        return y
    