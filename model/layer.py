import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron


class CustomEvaluator(torch.nn.Module):
    def __init__(self, evaluator: torch.nn.Module, model_type: str, model_time_step: int):
        super().__init__()
        self.evaluator = evaluator
        self.model_type = model_type
        self.model_time_step = model_time_step

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = self.evaluator(input, target)
        if self.model_type == "bptt":
            return loss
        elif self.model_type == "rate":
            loss = loss.detach() + (loss / self.model_time_step - (loss / self.model_time_step).detach())
            return loss
        else:
            return NotImplementedError()


class ReadOut(nn.Module):
    def __init__(self, model='avg'):
        super(ReadOut, self).__init__()

    def forward(self, spike):
        if self.step_mode == 's':
            return spike
        else:
            output = spike.reshape(self.time_step, -1, spike.shape[1])
            avg_fr = output.mean(dim=0)
            return avg_fr


class RateReadOut(nn.Module):
    def __init__(self, model='avg'):
        super(RateReadOut, self).__init__()

    def forward(self, x):
        if self.step_mode == 's':
            return x
        else:
            if self.training and self.step_mode == 'm' and not torch.is_grad_enabled():
                # spike prop
                output = x.reshape(self.time_step, -1, x.shape[1])
                avg_fr = output.mean(dim=0)
                return avg_fr
            elif self.training and self.step_mode == 'm' and torch.is_grad_enabled():
                # rate prop
                return x
            elif not self.training:
                # inference
                output = x.reshape(self.time_step, -1, x.shape[1])
                avg_fr = output.mean(dim=0)
                return avg_fr
            else:
                raise NotImplementedError()


class LIFLayer(neuron.LIFNode):

    def __init__(self, **cell_args):
        super(LIFLayer, self).__init__()
        tau = 1.0 / (1.0 - torch.sigmoid(cell_args['decay'])).item()
        super().__init__(tau=tau, decay_input=False, v_threshold=cell_args['thresh'], v_reset=cell_args['v_reset'],
                         detach_reset=cell_args['detach_reset'], step_mode='s')
        self.register_memory('elig', 0.)
        self.register_memory('elig_factor', 1.0)
        self.register_memory('out_spikes_mean', 0.)

    @staticmethod
    @torch.jit.script
    def calcu_sg_and_elig(current_t: int, v: torch.Tensor, elig: torch.Tensor, elig_factor: float, v_threshold: float,
                          sigmoid_alpha: float = 4.0):
        sgax = ((v - v_threshold) * sigmoid_alpha).sigmoid()
        sg = (1. - sgax) * sgax * sigmoid_alpha
        elig = 1. / (current_t + 1) * (current_t * elig + elig_factor * sg)
        return sg, elig

    def calcu_elig_factor(self, elig_factor, lam, sg, spike):
        if self.v_reset is not None:  # hard-reset
            elig_factor = self.calcu_elig_factor_hard_reset(elig_factor, lam, spike, self.v, sg)
        else:  # soft-reset
            if not self.detach_reset:  # soft-reset w/ reset_detach==False
                elig_factor = self.calcu_elig_factor_soft_reset_not_detach_reset(elig_factor, lam, sg)

            else:  # soft-reset w/ reset_detach==True
                elig_factor = self.calcu_elig_factor_soft_reset_detach_reset(elig_factor, lam)

        return elig_factor

    @staticmethod
    @torch.jit.script
    def calcu_elig_factor_hard_reset(elig_factor: torch.Tensor, lam: float, spike: torch.Tensor, v: torch.Tensor,
                                     sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam * (1. - spike) - lam * v * sg)
        return elig_factor

    @staticmethod
    @torch.jit.script
    def calcu_elig_factor_soft_reset_not_detach_reset(elig_factor: torch.Tensor, lam: float, sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam - lam * sg)
        return elig_factor

    @staticmethod
    def calcu_elig_factor_soft_reset_detach_reset(elig_factor: float, lam: float):
        elig_factor = 1. + elig_factor * lam
        return elig_factor

    def elig_init(self, x: torch.Tensor):
        self.elig = torch.zeros_like(x.data)
        self.elig_factor = 1.0

    def reset_state(self):
        self.reset()
        self.curr_time_step = 0

    def forward(self, x, **kwargs):

        if not self.training or not (hasattr(self, "rate_flag") and self.rate_flag):
            if self.step_mode == 's':
                self.v_float_to_tensor(x)
                self.neuronal_charge(x)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
            else:
                assert len(x.shape) in (2, 4)
                x = x.view(self.time_step, -1, *x.shape[1:])

                self.reset()
                # self.v = torch.zeros_like(x[0])
                spikes = []
                for t in range(self.time_step):
                    self.v_float_to_tensor(x[t])
                    self.neuronal_charge(x[t])
                    spike = self.neuronal_fire()
                    spikes.append(spike)
                    self.neuronal_reset(spike)

                # out = torch.stack(spikes, dim=0) if not self.train_mode_multi else torch.cat(spikes, dim=0)
                out = torch.cat(spikes, dim=0)
                return out
        elif self.training and not torch.is_grad_enabled() and self.step_mode == 's':
            assert len(x.shape) in (2, 4)

            lam = 1.0 - 1. / self.tau
            self.v_float_to_tensor(x)
            if isinstance(self.elig, float):
                self.elig = torch.full_like(x.data, self.elig)

            self.neuronal_charge(x)
            spike = self.neuronal_fire()

            t = self.curr_time_step
            sg, self.elig = self.calcu_sg_and_elig(current_t=t, v=self.v, elig=self.elig, elig_factor=self.elig_factor,
                                                   v_threshold=self.v_threshold)
            self.elig_factor = self.calcu_elig_factor(self.elig_factor, lam, sg, spike)

            if t == 0:
                self.out_spikes_mean = spike
            else:
                self.out_spikes_mean = 1. / (t + 1) * (t * self.out_spikes_mean + spike)

            self.neuronal_reset(spike)
            self.curr_time_step += 1

            return spike

        elif self.training and not torch.is_grad_enabled() and self.step_mode == 'm':
            assert len(x.shape) in (2, 4)
            x = x.view(self.time_step, -1, *x.shape[1:])

            self.reset()
            spikes = []

            lam = 1.0 - 1. / self.tau

            elig_factor = 1.0
            self.v_float_to_tensor(x[0])
            self.elig_init(x[0])
            for t in range(self.time_step):
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()

                sg, self.elig = self.calcu_sg_and_elig(current_t=t, v=self.v, elig=self.elig, elig_factor=elig_factor,
                                                       v_threshold=self.v_threshold)
                elig_factor = self.calcu_elig_factor(elig_factor, lam, sg, spike)
                spikes.append(spike)
                self.neuronal_reset(spike)
            out = torch.cat(spikes, dim=0)

            self.out_spikes_mean = out.view(self.time_step, -1, *out.shape[1:]).mean(dim=0)
            return out

        elif self.training and torch.is_grad_enabled():
            assert len(x.shape) in (2, 4)
            assert self.elig is not None and self.out_spikes_mean is not None
            rate = self.out_spikes_mean.detach() + (x * self.elig) - (x * self.elig).detach()
            return rate
        else:
            raise NotImplementedError()
