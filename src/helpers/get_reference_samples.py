from collections import defaultdict

import torch


def get_reference_samples_for_plot(dataset,
                                   n_digits_per_class,
                                   test_loader,
                                   device):
    xref_label, yref_label = None, None
    if dataset in ['mnist-mnist', 'mnist-svhn']:
        xrefs, yrefs = get_reference_samples_by_digit(test_loader, n_digits_per_class, device)
        xref = torch.cat([xrefs[digit] for digit in range(10)], dim=0)
        xref_label = torch.cat([digit * torch.ones(len(xrefs[digit])) for digit in range(10)], dim=0)
        yref = torch.cat([yrefs[digit] for digit in range(10)], dim=0)
        yref_label = torch.cat([digit * torch.ones(len(yrefs[digit])) for digit in range(10)], dim=0)
    elif dataset == 'mnist-cdcb':
        # assume batch_size > ncond, nref
        nref = 10 * n_digits_per_class
        batch = next(iter(test_loader))
        xref, yref = batch[0][:nref], batch[1][:nref]
        assert len(xref) == nref
    elif dataset == 'mnist-multiply':
        # TODO: implement label (not urgent)
        nref = 10 * n_digits_per_class
        batch = next(iter(test_loader))
        xref, yref = batch[0][0][:nref], batch[1][0][:nref]
        assert len(xref) == nref
    elif dataset in ['cub', 'cub-imgft2sent']:
        nref = 20
        batch = next(iter(test_loader))
        xref = batch[0][0][:nref] if dataset == 'cub' else batch[0][:nref]
        yref = batch[1][0][:nref]
        assert len(xref) == nref
    else:
        raise ValueError

    return xref.to(device), xref_label, yref.to(device), yref_label


def get_reference_samples_by_digit(test_loader, n_per_class, device):
    """For (MNIST, [MNIST, SVHN]) datasets

    Parameters
    ----------
    test_loader
    n_per_class

    Returns
    -------

    """
    xrefs = defaultdict(list)
    yrefs = defaultdict(list)
    counts = defaultdict(int)
    test_loader_iterator = iter(test_loader)
    while 1:
        try:
            batch = next(test_loader_iterator)
        except StopIteration:
            test_loader_iterator = iter(test_loader)
            batch = next(test_loader_iterator)

        for i in range(len(batch)):
            digit = batch[0][1][i].item()
            if len(xrefs[digit]) < n_per_class:
                xrefs[digit].append(batch[0][0][i])
                yrefs[digit].append(batch[1][0][i])
                counts[digit] += 1
                if all([counts[digit] >= n_per_class for digit in range(10)]):
                    break
        else:
            continue
        break

    for digit in range(10):
        xrefs[digit] = torch.stack(xrefs[digit], dim=0).to(device)
        yrefs[digit] = torch.stack(yrefs[digit], dim=0).to(device)

    return xrefs, yrefs
