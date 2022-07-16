import itertools

import numpy as np
import torch

from .ais import ais


def bdmc(model,
         loader,
         forward_schedule,
         n_sample,
         likelihood,
         likelihood_var,
         device=torch.device('cpu')):
    """Bidirectional Monte Carlo. Backward schedule is set to be the reverse of
    the forward schedule.

    Args:
      model (nn.Module): VAE model
        must have `latent_dim` attribute and `decode` method
      loader (iterator): iterator to loop over pairs of Variables; the first
        entry being `x`, the second being `z` sampled from the *true*
        posterior `p(z|x)`
      forward_schedule (list or numpy.ndarray): forward temperature schedule
      n_sample (int): number of importance samples
      likelihood
      likelihood_var
      device

    Returns:
        Two lists for forward and backward bounds on batches of data
    """

    # iterator is exhaustible in python 3, so need duplicate
    loader_forward, loader_backward = itertools.tee(loader, 2)

    # forward chain
    forward_logws, lower_bounds = ais(
        model,
        loader_forward,
        forward_schedule,
        n_sample,
        likelihood,
        likelihood_var,
        use_encoder=False,
        posterior_var=0.,
        forward=True,
        device=device)

    # backward chain
    backward_schedule = np.flip(forward_schedule, axis=0)
    backward_logws, upper_bounds = ais(
        model,
        loader_backward,
        backward_schedule,
        n_sample,
        likelihood,
        likelihood_var,
        use_encoder=False,
        posterior_var=0.,
        forward=False,
        device=device)

    print('Average bounds on simulated data: lower {:.4f}, upper {:.4f}'
          .format(lower_bounds, upper_bounds))

    return forward_logws, backward_logws
