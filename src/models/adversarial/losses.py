import numpy as np
import torch
import torch.nn.functional as F


def binary_entropy(p):
    return - p * np.log(p) - (1 - p) * np.log(1 - p)


# loss for density ratio estimation
# Note1: as a convention, r \approx p / q when the following losses are *maximized*
# Note2: In the following, **kwargs is for compatibility with the optional positive_label_smoothing in loss_js_var
def loss_js_var(lrp, lrq, positive_label_smoothing=1.):
    assert 0 < positive_label_smoothing <= 1
    a = positive_label_smoothing
    if a < 1:
        return (a * torch.log(a * torch.sigmoid(lrp))
                + (1 - a) * torch.log(1 - a * torch.sigmoid(lrp))
                + torch.log(1 - a * torch.sigmoid(lrq))
                + 2 * binary_entropy(a / 2)).mean()
    else:
        return (torch.log(torch.sigmoid(lrp))
                + torch.log(torch.sigmoid(-lrq))
                + 2 * binary_entropy(a / 2)).mean()


def loss_symmetric_kl_var_dv(lrp, lrq, **kwargs):
    return (lrp - lrq).mean() - torch.log(torch.exp(-lrp).mean()) - torch.log(torch.exp(lrq).mean())


def loss_symmetric_kl_var_njw(lrp, lrq, **kwargs):
    return (lrp - lrq - torch.exp(-lrp) - torch.exp(lrq)).mean() + 2


def loss_lc_var(lrp, lrq, **kwargs):
    return -2 * (torch.sigmoid(lrp) ** 2 + torch.sigmoid(-lrq) ** 2).mean() + 1


def loss_ratio_consistency(lrp, lrq):
    return (torch.exp(-lrp).mean() - 1) ** 2 + (torch.exp(lrq).mean() - 1) ** 2


# loss for model training
def loss_js_plugin(lrp, lrq):
    return loss_js_var(lrp, lrq)


def loss_symmetric_kl_plugin(lrp, lrq):
    # Note: if r = p / q, it becomes D(p||q) + D(q||p)
    return (lrp - lrq).mean()


def loss_reverse_kl_plugin(lrp, lrq):
    # if r = p / q, it is equivalent to D(q||p)
    return -lrq.mean()


def loss_forward_kl_plugin(lrp, lrq):
    # if r = p / q, it is equivalent to D(p||q)
    return lrp.mean()


def loss_l1(xh, x):
    return torch.abs(x - xh).mean()


def loss_l2(xh, x):
    return ((x - xh) ** 2).mean()


def loss_log_prob(logits, target):
    return F.cross_entropy(input=logits.view(-1, logits.shape[-1]),
                           target=target.long().view(-1),
                           reduction='mean',
                           ignore_index=1)  # do we need to multiply target.shape[-1]?


def loss_cos(x, xh):
    return 1 - F.cosine_similarity(x1=x, x2=xh, dim=-1).mean()  # cosine distance over embedding dimension


GENERATOR_LOSS_FUNCTIONS = {'js_plugin': loss_js_plugin,
                            'reverse_kl_plugin': loss_reverse_kl_plugin,
                            'forward_kl_plugin': loss_forward_kl_plugin,
                            'symmetric_kl_plugin': loss_symmetric_kl_plugin,
                            }

DISCRIMINATOR_LOSS_FUNCTIONS = {'js_var': loss_js_var,
                                'symmetric_kl_var_dv': loss_symmetric_kl_var_dv,
                                'symmetric_kl_var_njw': loss_symmetric_kl_var_njw,
                                'lc_var': loss_lc_var,
                                }

RECON_LOSS_FUNCTIONS = {'l1': loss_l1,
                        'l2': loss_l2,
                        'log': loss_log_prob,
                        'cos': loss_cos,
                        }
