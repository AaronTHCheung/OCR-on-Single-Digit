import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
from decimal import Decimal
from src.Models import CNN
import math
import argparse

if __name__ == "__main__":
    # Reading arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='Filename of the best trained CNN model')
    parser.add_argument('--batch', type=int, help='Training batch size')
    parser.add_argument('--epoch', type=int, help='Training epoches')
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch if args.batch is not None else 16  # training batch size
    epoches = args.epoch if args.epoch is not None else 100  # training epoches
    learning_rate = 1*torch.logspace(-1, -5, epoches)  # exponentially decaying learning rate to make the training faster

    # Paths
    model_filename = args.filename if args.filename is not None else 'cnn'  # filename for the best model
    mnist_path = 'data/'
    model_path = 'model/'
    model_filepath = model_path + model_filename + '.pth'

    # Loading MNIST data
    mndata = MNIST(mnist_path)
    train_x_1d, train_y = mndata.load_training()
    train_x_1d = torch.tensor(train_x_1d)/255
    train_x_2d = train_x_1d.view(train_x_1d.shape[0], 1, int(math.sqrt(train_x_1d.shape[1])), -1)
    train_y = F.one_hot(torch.tensor(train_y)).float()

    test_x_1d, test_y = mndata.load_testing()
    test_x_1d = torch.tensor(test_x_1d)/255
    test_x_2d = test_x_1d.view(test_x_1d.shape[0], 1, int(math.sqrt(test_x_1d.shape[1])), -1)
    test_y = F.one_hot(torch.tensor(test_y)).float()

    # Setting up datasets and dataloaders
    train_dataset_2d = torch.utils.data.TensorDataset(train_x_2d, train_y)
    test_dataset_2d = torch.utils.data.TensorDataset(test_x_2d, test_y)
    train_loader_2d = torch.utils.data.DataLoader(
        dataset=train_dataset_2d,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader_2d = torch.utils.data.DataLoader(
        dataset=test_dataset_2d,
        batch_size=batch_size,
        shuffle=True
    )

    # Initializing model
    model = CNN()
    train_loader = train_loader_2d
    test_loader = test_loader_2d
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    print('--------------Model Summary--------------')
    print(model)
    print("")

    # Training
    best_accuracy = 0
    for e in range(epoches):
        for p in optimizer.param_groups:
            p['lr'] = learning_rate[e]  # Decaying learning rate to speed up the training process

        for _, (x, y) in enumerate(train_loader):
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Testing
        correct_count = 0
        for _, (x, y) in enumerate(test_loader):
            y_hat = model(x)
            digit_hat = torch.max(input=y_hat.detach(), dim=1)[1]  # digit with the highest probability
            digit_target = torch.max(input=y.detach(), dim=1)[1]
            correct_count += (digit_hat == digit_target).sum().item()
        curr_accuracy = correct_count / test_y.shape[0]
        
        if (curr_accuracy > best_accuracy):  # Save the current model if it performs better than the previous best model
            best_accuracy = curr_accuracy
            torch.save({
                'state_dict': model.state_dict(),
                'accuracy': best_accuracy,
                'epoch': e
            }, model_filepath)
        print("Epoch (" + str(e+1) + "/"+str(epoches)+"): Testing Accuracy (current, best)=("+'%.3f' %
              Decimal(curr_accuracy)+", "+'%.3f' % Decimal(best_accuracy)+")")

    best_model_save = torch.load(model_filepath)
    model = CNN()
    model.load_state_dict(best_model_save['state_dict'])
