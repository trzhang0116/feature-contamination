import argparse
import os


def get_args(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        default=False,
        help='Evaluation only.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Training from an existing model.'
    )
    parser.add_argument(
        '--rd',
        action='store_true',
        default=False,
        help='Use representation distillation.'
    )
    parser.add_argument(
        '--lp',
        action='store_true',
        default=False,
        help='Use linear probing (True) or fine-tuning (False).'
    )
    parser.add_argument(
        '--distill-weight',
        type=float,
        default=0.9,
        help='Weight for distillation loss.'
    )
    parser.add_argument(
        '--attn-init',
        action='store_true',
        default=False,
        help='Use CLIP-pretrained weights to initialize attention layers in distillation.'
    )
    parser.add_argument(
        '--attn-distill',
        action='store_true',
        default=False,
        help='Use attention distillation when distilling ViTs.'
    )
    parser.add_argument(
        '--attn-distill-weight',
        type=float,
        default=0.1,
        help='Weight for attention distillation loss.'
    )
    parser.add_argument(
        '--attn-distill-blocks',
        type=lambda x: x.split(','),
        default='last',
        help='Indices of distilled attention blocks in attention distillation. Use "last" or "-1" to distill the last block.'
    )
    parser.add_argument(
        '--layer-distill',
        action='store_true',
        default=False,
        help='Use layer-wise distillation of hidden states when distilling ViTs.'
    )
    parser.add_argument(
        '--layer-distill-weight',
        type=float,
        default=0.1,
        help='Weight for layer-wise distillation loss.'
    )
    parser.add_argument(
        '--layer-distill-blocks',
        type=lambda x: x.split(','),
        default='-1',
        help='Indices of distilled hidden layers in layer-wise distillation. Use "last" or "-1" to distill the last hidden layer.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./datasets',
        help='The root directory of datasets.'
        )
    parser.add_argument(
        '--load-dir',
        type=str,
        default=None,
        help='The root directory for loading models; None means not to load.'
    )
    parser.add_argument(
        '--classifier-load-dir',
        type=str,
        default=None,
        help='The root directory for loading linear classifiers; None means not to load.'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='./results',
        help='The root directory for storing results and models; None means not to store.'
    )
    parser.add_argument(
        '--feature-cache-dir',
        type=str,
        default=None,
        help='The root directory for storing extracted features.'
    )
    parser.add_argument(
        '--train-dataset',
        default='ImageNet',
        type=str,
        help='Dataset used for training.'
    )
    parser.add_argument(
        '--eval-datasets',
        default='ImageNet,ImageNetV2,ImageNetR,ObjectNet,ImageNetSketch,ImageNetA',
        type=lambda x: x.split(','),
        help='Datasets used for evaluation; split by comma.'
    )
    parser.add_argument(
        '--zeroshot-init',
        action='store_true',
        default=False,
        help='Use zero-shot classification head initialization.'
    )
    parser.add_argument(
        '--template',
        type=str,
        default=None,
        help='Prompt template used for initializing the linear classifier.'
    )
    parser.add_argument(
        '--classnames',
        type=str,
        default="openai",
        help='Class names used in the prompts.'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        default='test',
        help='Name of the experiment.'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='CLIP_RN50',
        help='The name of the model; e.g., ERM_ResNet50_V2, CLIP_RN50x16.'
    )
    parser.add_argument(
        '--reversible',
        action='store_true',
        default=False,
        help='Use reversible MLP for feature extraction.'
    )
    parser.add_argument(
        '--residual',
        action='store_true',
        default=False,
        help='Use residual blocks for feature extraction.'
    )
    parser.add_argument(
        '--n-nonlinear-transform-blocks',
        type=int,
        default=0,
        help='The number of blocks used in non-linear feature transformation.'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=0,
        help='The number of layers of the featurizer; 0 means linear probing.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate.'
    )
    parser.add_argument(
        '--lr-warm-restarts',
        action='store_true',
        default=False,
        help='Use cosine annealing with warm restarts for learning rate scheduling.'
    )
    parser.add_argument(
        '--restart-epochs',
        action='store_true',
        default=False,
        help='Number of epochs performed before each restart in warm restarts.'
    )
    parser.add_argument(
        '--oracle-norm-stats',
        action='store_true',
        default=False,
        help='Use oracle normalization statistics from CLIP for all normalization layers in distillation.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='L2 weight decay.'
    )
    parser.add_argument(
        '--ls',
        type=float,
        default=0.0,
        help='Label smoothing.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=32,
        help='The number of workers for data loading.'
    )
    parser.add_argument(
        '--pin-memory',
        action='store_true',
        default=False,
        help='Use pinned memory when loading data.'
    )
    parser.add_argument(
        '--warmup-length',
        type=int,
        default=500,
        help='The number of gradient steps in warmup.'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=1000,
        help='The frequency of printing training info.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID to use.'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='Use distributed training.'
    )
    parser.add_argument(
        '--world-size',
        default=-1,
        type=int,
        help='Number of nodes for distributed training.'
    )
    parser.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='Node rank for distributed training.'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='URL used to set up distributed training.'
    )
    parser.add_argument(
        '--dist-backend',
        default='nccl',
        type=str,
        help='Distributed backend.'
    )

    args = parser.parse_args()

    if verbose:
        print('Args:')
        for k, v in vars(args).items():
            print('\t{}: {}'.format(k, v))
    
    return args
    