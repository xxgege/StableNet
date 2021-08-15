import argparse

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-data', metavar='DIR', default='/DATA/DATANAS1/windxrz/dataset/PACS/split_compositional_with_val_sketch',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_with_table',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=3, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log_base',
                    default='./results', type=str, metavar='PATH',
                    help='path to save logs (default: none)')

# for number of fourier spaces
parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')

parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')
parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')

# parser.add_argument ('--cfs', type = int, default = 512, help = 'the dim of each feature')
parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument ('--epochp', type = int, default = 0, help = 'number of epochs to pretrain')

parser.add_argument ('--n_feature', type=int, default=128, help = 'number of pre-saved features')
parser.add_argument ('--feature_dim', type=int, default=512, help = 'the dim of each feature')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

# for jointly train
parser.add_argument ('--train_cnn_with_lossb', type=bool, default=False, help = 'whether train cnn with lossb')
parser.add_argument ('--cnn_lossb_lambda', type=float, default=0, help = 'lambda for lossb')

# for more moments
parser.add_argument ('--moments_lossb', type=float, default=1, help = 'number of moments')

# for first step
parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

# for pow
parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for second order moment weight
parser.add_argument ('--second_lambda', type=float, default=0.2, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.05, help = 'weight lambda for second order moment loss')

# for dat/.a aug
parser.add_argument ('--lower_scale', type=float, default=0.8, help = 'weight lambda for second order moment loss')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')

parser.add_argument ('--classes_num', type=int, default=5, help = 'number of epoch for lambda to decay')

parser.add_argument ('--dataset', type=str, default="PACS", help = '')
parser.add_argument ('--sub_dataset', type=str, default="", help = '')
parser.add_argument ('--gray_scale', type=float, default=0.1, help = 'weight lambda for second order moment loss')

parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=1, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
