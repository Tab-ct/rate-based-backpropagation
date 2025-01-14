import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from .layer import RateReadOut


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
            in_spikes_mean, in_spikes_var = calcu_spikes_mean_and_var(args[0])

            curr_t = module.curr_time_step

            module.in_spikes_mean = 1. / (curr_t + 1) * (
                    curr_t * module.in_spikes_mean + in_spikes_mean) if module.in_spikes_mean is not None else in_spikes_mean

            in_spikes_var = in_spikes_var + module.eps
            module.in_spikes_var_recip = 1. / (curr_t + 1) * (
                    curr_t * module.in_spikes_var_recip + 1. / in_spikes_var) if module.in_spikes_var is not None else 1. / in_spikes_var
        else:
            assert len(args[0].shape) == 4
            in_spikes_mean, in_spikes_var = calcu_spikes_mean_and_var(args[0])

            module.in_spikes_mean = in_spikes_mean
            module.in_spikes_var = in_spikes_var

    else:  # rate_prop stage
        if module.step_mode == 's':
            assert len(args[0].shape) == 4
            assert module.training

            rate = calcu_rate(args[0], module.in_spikes_mean, 1. / module.in_spikes_var_recip,
                              gamma=module.weight.view(1, module.weight.shape[0], 1, 1),
                              beta=module.bias.view(1, module.bias.shape[0], 1, 1), eps=module.eps)

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

    # rate prop stage
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
            if isinstance(module, (nn.BatchNorm2d,)):
                module.in_spikes_mean = None
                module.in_spikes_var = None
                module.in_spikes_var_recip = None


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
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
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
