import os
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=None, type=int)

# model setting
parser.add_argument('--arch', default="vggsnn", type=str, help="vggsnn")
parser.add_argument('--resume', default=None, type=str)

# input data preprocess
parser.add_argument('--dataset', default="CIFAR10_DVS_Aug", type=str, help="dataset")
parser.add_argument('--data_path', default="[your data path]", type=str)
parser.add_argument('--log_path', default="./log", type=str, help="log path")
parser.add_argument('--auto_aug', default=False, action='store_true')
parser.add_argument('--cutout', default=False, action='store_true')

# learning setting
parser.add_argument('--optim', default='SGDM', type=str)
parser.add_argument('--scheduler', default='COSINE', type=str)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--val_batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--num_workers', default=0, type=int)

parser.add_argument('--decay', default=0.2, type=float)
parser.add_argument('--v_reset', default=None, type=float)
parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--T', default=10, type=int, help='num of time steps')
parser.add_argument('--step_mode', choices=['s', 'm'], help='step mode')
parser.add_argument('--detach_reset', default=True, action='store_true')

# training algorithm
parser.add_argument('--rate_flag', default=False, action='store_true')

args = parser.parse_args()
