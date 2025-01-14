import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=None, type=int, help='random seed')

# model setting
parser.add_argument('--arch', default="sew_resnet34", type=str, help="sew_resnet34|preact_resnet34")
parser.add_argument('--resume', default=None, type=str, help='pth file that holds the model parameters')

# input data preprocess
parser.add_argument('--dataset', default="imagenet", type=str, help="CIFAR10|CIFAR100")
parser.add_argument('--data_path', default="your data path", type=str)
parser.add_argument('--log_path', default="./log", type=str, help="log path")
parser.add_argument('--auto_aug', default=False, action='store_true')
parser.add_argument('--cutout', default=False, action='store_true')

# learning setting
parser.add_argument('--optim', default='SGDM', type=str)
parser.add_argument('--scheduler', default='COSINE', type=str)
parser.add_argument('--train_batch_size', default=512, type=int)
parser.add_argument('--val_batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.2, type=float)
parser.add_argument('--wd', default=2e-5, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--num_workers', default=16, type=int)

# spiking neuron setting
parser.add_argument('--decay', default=0.2, type=float)
parser.add_argument('--v_reset', default=None, type=float)
parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--T', default=4, type=int, help='num of time steps')
parser.add_argument('--step_mode', choices=['s', 'm'], help='step mode')
parser.add_argument('--detach_reset', default=True, action='store_true')

# training algorithm
parser.add_argument('--rate_flag', default=False, action='store_true')
parser.add_argument("--local-rank", default=-1)

args = parser.parse_args()
