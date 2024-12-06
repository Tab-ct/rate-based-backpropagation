import json
import numpy as np
import random
import logging
import os

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional

from model.layer import RateReadOut


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


@torch.jit.script
def calcu_spikes_mean_and_var(spike: torch.Tensor):
    in_spikes_mean = spike.mean(dim=(0, 2, 3), keepdim=True)
    in_spikes_var = ((spike - in_spikes_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    return in_spikes_mean, in_spikes_var


@torch.jit.script
def calcu_rate(x: torch.Tensor, in_spikes_mean: torch.Tensor, in_spikes_var: torch.Tensor, gamma: torch.Tensor,
               beta: torch.Tensor, eps: float):
    rate_mean = x.mean(dim=(0, 2, 3), keepdim=True)
    rate_mean = in_spikes_mean.detach() + (rate_mean - rate_mean.detach())

    rate_var = ((x - rate_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    rate_var = in_spikes_var.detach() + (rate_var - rate_var.detach())
    rate_var = nn.functional.relu(rate_var)

    rate_hat = (x - rate_mean) / torch.sqrt((rate_var + eps))
    rate = gamma * rate_hat + beta
    return rate


def bn_forward_hook(module, args, output):
    """Batch Normalization 2D"""

    # testing stage
    if not module.training:
        return

    # training stage
    if not torch.is_grad_enabled():  # spike_prop stage
        if module.step_mode == 's':
            module.in_spikes.append(args[0])
        else:
            assert len(args[0].shape) == 4
            in_spikes_mean, in_spikes_var = calcu_spikes_mean_and_var(args[0])

            module.in_spikes_mean = in_spikes_mean
            module.in_spikes_var = in_spikes_var

    else:  # rate_prop stage
        if module.step_mode == 's':
            assert module.in_spikes is not None and len(module.in_spikes) == module.time_step
            rate = args[0]
            out_spikes = []
            for t in range(module.time_step):
                x = module.in_spikes[t].detach() + rate - rate.detach()
                x = module.forward(x)
                out_spikes.append(x)
            rate = torch.stack(out_spikes, dim=0).mean(dim=0)

            module.track_running_stats = True
            return rate

        else:
            assert len(args[0].shape) == 4
            assert module.training

            rate = calcu_rate(args[0], module.in_spikes_mean, module.in_spikes_var,
                              gamma=module.weight.view(1, module.weight.shape[0], 1, 1),
                              beta=module.bias.view(1, module.bias.shape[0], 1, 1), eps=module.eps)

            module.track_running_stats = True
            return rate


def bn_forward_pre_hook(module, input):
    # testing stage or spike_prop stage
    if not torch.is_grad_enabled():
        return

    # rate_prop stage
    module.track_running_stats = False
    return


def model_forward_pre_hook(module, input):
    if not module.training:
        return

    torch.set_grad_enabled(False)


def model_forward_hook(module, args, output):
    if not module.training:
        return

    if not hasattr(module, "out_spikes"):
        module.out_spikes = []

    if module.step_mode == 's':
        # record grad in spikes[0:T-1]
        output.requires_grad = True
        module.out_spikes.append(output)
        module.curr_time_step += 1
        if module.curr_time_step != module.time_step:
            return

        # rate prop
        torch.set_grad_enabled(True)
        out_rate = module.forward(args[0])

        # replace rate with spike in BP graph
        assert len(module.out_spikes) == module.time_step
        surrogate_output = output.detach() + out_rate - out_rate.detach()
        return surrogate_output
    else:
        # rate_prop
        torch.set_grad_enabled(True)
        in_data = args[0]
        in_data = in_data.reshape(module.time_step, -1, *in_data.shape[1:])
        in_rate = in_data.mean(dim=0).detach()
        out_rate = module.forward(in_rate)

        return out_rate


def model_backward_pre_hook(module, grad_output):
    if module.step_mode == 'm':
        return
    elif module.step_mode == 's':
        assert hasattr(module, "out_spikes") and len(module.out_spikes) == module.time_step
        total_grad = sum(tensor.grad for tensor in module.out_spikes[:-1]) + grad_output[0]
        module.out_spikes.clear()
        return total_grad,
    else:
        raise NotImplementedError()


def model_backward_hook(module, grad_input, grad_output):
    init_model(module)


def init_model(model):
    functional.reset_net(model)

    if model.step_mode == 's':
        for name, module in model.named_modules():
            module.curr_time_step = 0

    if hasattr(model, "rate_hooks") and isinstance(model.rate_hooks, list) and len(model.rate_hooks) > 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.in_spikes = []
                module.in_spikes_mean = None
                module.in_spikes_var = None


def bptt_model_setting(model: nn.Module, **kwargs):
    assert ('time_step' in kwargs and kwargs.get('time_step') > 0) and 'step_mode' in kwargs
    time_step, step_mode = kwargs.get('time_step'), kwargs.get('step_mode')

    for name, module in model.named_modules():
        setattr(module, 'time_step', time_step)
        setattr(module, 'step_mode', step_mode)

    init_model(model)


def rate_model_setting(model: nn.Module, **kwargs):
    assert ('time_step' in kwargs and kwargs.get('time_step') > 0) and 'step_mode' in kwargs
    time_step, step_mode = kwargs.get('time_step'), kwargs.get('step_mode')

    assert hasattr(model, "readout")
    model.bptt_readout = model.readout
    model.readout = RateReadOut()

    for name, module in model.named_modules():
        setattr(module, 'time_step', time_step)
        setattr(module, 'step_mode', step_mode)

    model.rate_hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_hook01 = module.register_forward_pre_hook(bn_forward_pre_hook)
            bn_hook02 = module.register_forward_hook(bn_forward_hook)
            model.rate_hooks.extend([bn_hook01, bn_hook02])
        elif isinstance(module, neuron.BaseNode):
            module.rate_flag = True
        else:
            continue

    hook01 = model.register_forward_pre_hook(model_forward_pre_hook)
    hook02 = model.register_forward_hook(model_forward_hook)
    hook03 = model.register_full_backward_hook(model_backward_hook)
    hook04 = model.register_full_backward_pre_hook(model_backward_pre_hook)
    model.rate_hooks.extend([hook01, hook02, hook03, hook04])

    init_model(model)


def restore2rate_model(model: nn.Module):
    assert hasattr(model, "rate_hooks") and isinstance(model.rate_hooks, list)
    for handle in model.rate_hooks:
        handle.remove()

    for name, module in model.named_modules():
        if isinstance(module, neuron.BaseNode) and hasattr(module, "rate_flag"):
            del module.rate_flag

    model.readout = model.bptt_readout
