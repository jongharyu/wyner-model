import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datasets.dataloaders import get_dataloaders_mnist, get_dataloaders_svhn
from models.classifiers.model import MNISTClassifier, SVHNClassifier


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        X = 2 * X - 1
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def evaluate(dataloader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for X, y_true in dataloader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        X = 2 * X - 1
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = evaluate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    # plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot_grid_samples(train_losses, color='blue', label='Training loss')
    ax.plot_grid_samples(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')


def get_accuracy(model, dataloader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0.
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in dataloader:
            X = X.to(device)
            y_true = y_true.to(device)

            X = 2 * X - 1
            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum().item()

    return correct_pred / n


def main(config):
    # check device
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # parameters
    random_seed = config.seed
    lr = config.lr
    batch_size = config.batch_size
    n_epochs = config.n_epochs
    dataset = config.dataset

    # download and create datasets
    main_path = config.main_path

    # define the data loaders
    if dataset == 'mnist':
        classifier = MNISTClassifier
        train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size, device=device, root_path=main_path,
                                                                        valid_split=0.1)
    elif dataset == 'svhn':
        classifier = SVHNClassifier
        train_loader, valid_loader, test_loader = get_dataloaders_svhn(batch_size, device=device, root_path=main_path,
                                                                       valid_split=0.1)
    else:
        raise ValueError

    torch.manual_seed(random_seed)

    model = classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, n_epochs, device)
    torch.save(model.state_dict(), '{}/{}.rar'.format(main_path, dataset))

    print("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning classifiers')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'svhn'])
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--main-path', type=str, default="..", help='main path where datasets live and loggings are saved')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA use')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for data (default: 32)')
    parser.add_argument('--n-epochs', type=int, default=15, metavar='E', help='number of epochs to train (default: 15)')

    config = parser.parse_args()
    main(config)
