import torch


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, dataset=None, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if dataset in ['mnist-cdcb', 'sketchy-vgg']:
        return [dataB[0].to(device), dataB[1].to(device)]
    if dataset == 'celeba':
        return [dataB[0].to(device), dataB[1].type(torch.FloatTensor).to(device)]
    if dataset == 'mnist-svhn':
        if is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]
        else:
            return [dataB[0].to(device), dataB[1].to(device)]

    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-mnist, mnist-svhn, cubIS

        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))

    elif torch.is_tensor(dataB):
        return dataB.to(device)

    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))


def unpack_data_for_classifier(dataB, device, mode):
    assert mode in ['xy2z', 'x2z', 'y2z']
    if len(dataB[0]) == 2:
        if mode == 'xy2z':
            return (dataB[0][0].to(device), dataB[1][0].to(device)), dataB[1][1].to(device)
        elif mode == 'x2z':
            return dataB[0][0].to(device), dataB[0][1].to(device)
        elif mode == 'y2z':
            return dataB[1][0].to(device), dataB[1][1].to(device)
    else:
        return dataB.to(device)