from __future__ import print_function

import numpy as np
import torch
from tqdm import tqdm

from .hmc import hmc_trajectory, accept_reject
from .utils import log_normal, log_bernoulli, log_mean_exp, safe_repeat


def ais(model,
        loader,
        forward_schedule,
        n_importance_sample,
        likelihood,
        likelihood_var,
        use_encoder,
        posterior_var,
        forward=True,
        n_batches=np.inf,
        device='cpu'):
    """Annealed importance sampling

    Args:
        model (nn.Module): deocder-based model
            must have `latent_dim` attribute and `decode` method
        loader (iterator): iterator to loop over pairs of Variables; the first
            entry being `x`, the second being `z` sampled from the *true*
            posterior `p(z|x)`
        forward_schedule (list or numpy.ndarray): forward temperature schedule
        n_importance_sample (int): number of importance samples
        likelihood
        likelihood_var
        use_encoder
        posterior_var
        forward
        device

    Returns:
        The list for (forward) bounds on batches of data and its average
    """
    # forward chain (for lower bounds on log w)
    forward_log_ws = ais_trajectory(
        model,
        loader,
        forward=forward,
        schedule=forward_schedule,
        n_importance_sample=n_importance_sample,
        likelihood=likelihood,
        likelihood_var=likelihood_var,
        use_encoder=use_encoder,
        posterior_var=posterior_var,
        n_batches=n_batches,
        device=device)  # length of n_batch list; each element is a tensor of size (batch_size,)

    lower_bounds = []

    for i, forward in enumerate(forward_log_ws):
        lower_bounds.append(forward.mean().detach().item())

    lower_bounds = np.mean(lower_bounds)

    return forward_log_ws, lower_bounds


def ais_trajectory(model,
                   loader,
                   forward=True,
                   schedule=None,
                   n_importance_sample=100,
                   likelihood='bernoulli',
                   likelihood_var=0.075,
                   use_encoder=False,
                   posterior_var=0.075,
                   n_batches=np.inf,
                   device='cpu'):
    """Compute annealed importance sampling trajectories for a batch of data.
    Could be used for *both* forward and reverse chain in BDMC.

    Args:
        model (torch.Model): decoder model
        loader (iterator): iterator that returns pairs, with first component
          being `x`, second would be `z` or label (will not be used)
        forward (boolean): indicate forward/backward chain
        schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^\tau`
        n_importance_sample (int): number of importance samples (or number of independent AIS weights)
        likelihood (str): either 'bernoulli' or 'normal'
        likelihood_var (float): likelihood variance for Gaussian observation model
        use_encoder (bool): indicate whether we use variational posterior or not
        posterior_var (float): posterior variance
        device (str or None)

    Returns:
        A list where each element is a torch.autograd.Variable that contains the
        log importance weights for a single batch of data
    """
    assert likelihood in ['normal', 'bernoulli']

    latent_dim = model.latent_dim

    def log_ft(z, data_x, post_z, tau):
        r"""Unnormalized density for intermediate distribution `f_t`:
            if use_encoder is True:
                f_t = q(z|x)^(1-\tau) (p(x,z) / q(z|x))^(\tau) = q(z|x) (p(z) p(x|z) / q(z|x))^\tau
                =>  log f_t = (1-\tau) * log q(z|x) + \tau * (log p(x|z) + \tau * log p(z))
            else:
                f_t = p(z)^(1-\tau) p(x,z)^(\tau) = p(z) p(x|z)^\tau
                =>  log f_t = log p(z) + \tau * log p(x|z)

        Parameters
        ----------
        z
        data_x
        post_z
            not used if use_encoder is False
        tau

        Returns
        -------
        log_ft
        """
        xhat = model.decode(z)
        if likelihood == 'normal':
            log_likelihood = log_normal(data_x, xhat, np.log(likelihood_var))
        elif likelihood == 'bernoulli':
            log_likelihood = log_bernoulli(xhat, data_x)
        else:
            raise ValueError
        log_prior = log_normal(z, 0., 0.)
        if use_encoder:
            log_posterior = log_normal(z, post_z, np.log(posterior_var))
            return log_posterior + tau * (log_prior + log_likelihood)
        else:
            return log_prior + tau * log_likelihood

    log_ws = []
    for i, (data_x, post_z) in enumerate(loader):
        if i > n_batches:
            break
        B = data_x.size(0) * n_importance_sample
        data_x = safe_repeat(data_x, n_importance_sample).to(device)
        post_z = safe_repeat(post_z, n_importance_sample).to(device)

        with torch.no_grad():
            epsilon = torch.ones(B).to(device).mul_(0.01)
            accept_hist = torch.zeros(B).to(device)
            log_w = torch.zeros(B).to(device)

        # initial sample of z
        if forward:
            current_z = torch.randn(B, latent_dim).to(device)
        else:
            current_z = post_z

        ones = torch.ones(B).to(device)
        for j, (tau0, tau1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            # 1) update log importance weight
            log_int_1 = log_ft(current_z, data_x, post_z, tau0).detach()
            log_int_2 = log_ft(current_z, data_x, post_z, tau1).detach()
            log_w += log_int_2 - log_int_1

            # 2) apply HMC transition kernel
            # resample velocity
            current_v = torch.randn(current_z.size()).to(device)

            def U(z):
                return -log_ft(z, data_x, post_z, tau1)

            def grad_U(z):
                z = z.detach().requires_grad_()
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = ones
                # torch.autograd.grad default returns volatile
                grad = torch.autograd.grad(U(z), z, grad_outputs=grad_outputs)[0]
                # clip by norm
                max_ = B * latent_dim * 100.
                grad = torch.clamp(grad, -max_, max_)
                return grad

            def normalized_kinetic(v):
                return -log_normal(v, 0., 0.)

            z, v = hmc_trajectory(current_z, current_v, grad_U, epsilon)
            current_z, epsilon, accept_hist = accept_reject(
                current_z, current_v,
                z, v,
                epsilon,
                accept_hist, j,
                U, K=normalized_kinetic,
                device=device)

        log_w = log_mean_exp(log_w.view(n_importance_sample, -1).transpose(0, 1))  # (batch_size,)
        if not forward:
            log_w = -log_w
        log_ws.append(log_w.data)
        print('{}-th (out of {}) batch {:.4f}'.format(i, n_batches, log_w.mean().cpu().data.numpy()))

    return log_ws
