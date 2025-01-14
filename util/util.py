import json
import numpy as np
import random
import logging
import os

import torch
from spikingjelly.activation_based import neuron, functional


class Logger:
    def __init__(self, args, log_path, write_file=True):
        self.log_path = log_path
        self.logger = logging.getLogger('')
        if write_file:
            filename = os.path.join(self.log_path, 'train.log')
            # file handler
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(message)s'))

        self.logger.setLevel(logging.INFO)
        if write_file:
            self.logger.addHandler(handler)
            self.logger.info("Logger created at {}".format(filename))
        self.logger.addHandler(console)

    def debug(self, strout):
        return self.logger.debug(strout)

    def info(self, strout):
        return self.logger.info(strout)

    def info_config(self, config):
        self.info('The hyperparameter list:')
        for k, v in vars(config).items():
            self.info('  --' + k + ' ' + str(v))

    def info_args(self, args):
        args_json = json.dumps(vars(args))
        self.info(args_json)


def setup_seed(seed):
    import os
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def split_params(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isinstance(module, neuron.LIFNode) and hasattr(module, "thresh"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            paras[1].append(module.weight)
            if module.bias is not None:
                paras[2].append(module.bias)
        elif len(list(module.children())) > 0:
            paras = split_params(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


def get_model_name(model_name, args):
    aug_str = '_'.join(['cut' if args.cutout else ''] + ['aug' if args.auto_aug else ''])
    if aug_str[0] != '_': aug_str = '_' + aug_str
    if aug_str[-1] != '_': aug_str = aug_str + '-'
    model_name += args.dataset.lower() + aug_str + 'snn' + '_t' + str(
        args.T) + '_' + args.arch.lower() + '_opt_' + args.optim.lower() + '_wd_' + str(args.wd)
    cas_num = len([one for one in os.listdir(args.log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)

    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        model_name += str(dist.get_rank())

    return model_name


def init_config(args):
    seed = setup_seed(args.seed)
    args.seed = seed

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    model_name = get_model_name('', args)
    args.log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)


def warp_decay(decay):
    import math
    return torch.tensor(math.log(decay / (1 - decay)))
