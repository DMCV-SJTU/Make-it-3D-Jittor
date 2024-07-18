import jittor as jt
from jittor import init
from jittor import nn
import math
from typing import List
from jittor.optim import Optimizer
from jittor import Var
import jtorch as torch
class Adan(Optimizer):
    "\n    Implements a pytorch variant of Adan\n    Adan was proposed in\n    Adan: Adaptive Nesterov Momentum Algorithm for\n        Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.\n    https://arxiv.org/abs/2208.06677\n    Arguments:\n        params (iterable): iterable of parameters to optimize or\n            dicts defining parameter groups.\n        lr (float, optional): learning rate. (default: 1e-3)\n        betas (Tuple[float, float, flot], optional): coefficients used for\n            first- and second-order moments. (default: (0.98, 0.92, 0.99))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability. (default: 1e-8)\n        weight_decay (float, optional): decoupled weight decay\n            (L2 penalty) (default: 0)\n        max_grad_norm (float, optional): value used to clip\n            global grad norm (default: 0.0 no clip)\n        no_prox (bool): how to perform the decoupled weight decay\n            (default: False)\n        foreach (bool): if True would use torch._foreach implementation.\n            It's faster but uses slightly more memory. (default: True)\n    "

    def __init__(self, params, lr=0.001, betas=(0.98, 0.92, 0.99), eps=1e-08, weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool=True):
        if (not (0.0 <= max_grad_norm)):
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if (not (0.0 <= lr)):
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if (not (0.0 <= eps)):
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if (not (0.0 <= betas[0] < 1.0)):
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if (not (0.0 <= betas[1] < 1.0)):
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if (not (0.0 <= betas[2] < 1.0)):
            raise ValueError('Invalid beta parameter at index 2: {}'.format(betas[2]))
        self.default = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params,lr)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)


    def restart_opt(self):
        with jt.no_grad():
            for group in self.param_groups:
                group['step'] = 0
                for p in group['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['exp_avg_diff'] = torch.zeros_like(p)


    def step(self, closure=None):
        with jt.no_grad():
            loss = None
            if (closure is not None):
                with torch.enable_grad():
                    loss = closure()
            if (self.default['max_grad_norm'] > 0):
                device = self.param_groups[0]['params'][0].device
                global_grad_norm = torch.zeros(1, device=device)
                max_grad_norm = torch.tensor(self.default['max_grad_norm'], device=device)
                for group in self.param_groups:
                    for p in group['params']:
                        if (p.grad is not None):
                            grad = p.grad
                            global_grad_norm.add_(grad.pow(2).sum())
                global_grad_norm = torch.sqrt(global_grad_norm)
                clip_global_grad_norm = torch.clamp(
                    max_grad_norm / (global_grad_norm + self.default['eps']),
                    max=1.0).item()
            else:
                clip_global_grad_norm = 1.0
            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                exp_avg_diffs = []
                neg_pre_grads = []
                (beta1, beta2, beta3) = self.default['betas']
                if ('step' in group):
                    group['step'] += 1
                else:
                    group['step'] = 1
                bias_correction1 = (1.0 - (beta1 ** group['step']))
                bias_correction2 = (1.0 - (beta2 ** group['step']))
                bias_correction3 = (1.0 - (beta3 ** group['step']))
                for p in group['params']:
                    if (p.grad is None):
                        continue
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if (len(state) == 0):
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['exp_avg_diff'] = torch.zeros_like(p)
                    if (('neg_pre_grad' not in state) or (group['step'] == 1)):
                        state['neg_pre_grad'] = p.grad.clone().mul_((- clip_global_grad_norm))
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avg_diffs.append(state['exp_avg_diff'])
                    neg_pre_grads.append(state['neg_pre_grad'])
                kwargs = dict(params=params_with_grad, grads=grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, exp_avg_diffs=exp_avg_diffs, neg_pre_grads=neg_pre_grads, beta1=beta1, beta2=beta2, beta3=beta3, bias_correction1=bias_correction1, bias_correction2=bias_correction2, bias_correction3_sqrt=math.sqrt(bias_correction3), lr=self.default['lr'], weight_decay=self.default['weight_decay'], eps=self.default['eps'], no_prox=self.default['no_prox'], clip_global_grad_norm=clip_global_grad_norm)
                if self.default['foreach']:
                    _multi_tensor_adan(**kwargs)
                else:
                    _single_tensor_adan(**kwargs)
            return loss

def _single_tensor_adan(params: List[Var], grads: List[Var], exp_avgs: List[Var], exp_avg_sqs: List[Var], exp_avg_diffs: List[Var], neg_pre_grads: List[Var], *, beta1: float, beta2: float, beta3: float, bias_correction1: float, bias_correction2: float, bias_correction3_sqrt: float, lr: float, weight_decay: float, eps: float, no_prox: bool, clip_global_grad_norm: Var):
    for (i, param) in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]
        grad.mul_(clip_global_grad_norm)
        neg_grad_or_diff.add_(grad)
        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
        exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha=(1 - beta2))
        neg_grad_or_diff.mul_(beta2).add_(grad)
        exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff, neg_grad_or_diff, value=(1 - beta3))
        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        step_size_diff = ((lr * beta2) / bias_correction2)
        step_size = (lr / bias_correction1)
        if no_prox:
            param.mul_((1 - (lr * weight_decay)))
            param.addcdiv_(exp_avg, denom, value=(- step_size))
            param.addcdiv_(exp_avg_diff, denom, value=(- step_size_diff))
        else:
            param.addcdiv_(exp_avg, denom, value=(- step_size))
            param.addcdiv_(exp_avg_diff, denom, value=(- step_size_diff))
            param.div_((1 + (lr * weight_decay)))
        neg_grad_or_diff.zero_().add_(grad, alpha=(- 1.0))

def _multi_tensor_adan(params: List[Var], grads: List[Var], exp_avgs: List[Var], exp_avg_sqs: List[Var], exp_avg_diffs: List[Var], neg_pre_grads: List[Var], *, beta1: float, beta2: float, beta3: float, bias_correction1: float, bias_correction2: float, bias_correction3_sqrt: float, lr: float, weight_decay: float, eps: float, no_prox: bool, clip_global_grad_norm: Var):
    if (len(params) == 0):
        return
    torch._foreach_mul_(grads, clip_global_grad_norm)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=(1 - beta1))
    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=(1 - beta2))
    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=(1 - beta3))
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)
    step_size_diff = ((lr * beta2) / bias_correction2)
    step_size = (lr / bias_correction1)
    if no_prox:
        torch._foreach_mul_(params, (1 - (lr * weight_decay)))
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=(- step_size))
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=(- step_size_diff))
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=(- step_size))
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=(- step_size_diff))
        torch._foreach_div_(params, (1 + (lr * weight_decay)))
    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=(- 1.0))
