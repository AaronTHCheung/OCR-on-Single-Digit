import torch.nn as nn
import torch.nn.functional as F

input_channels = 28*28  # MNIST images' dimension is 28*28
output_channels = 10  # 10 digit classification

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # construct layers
        self.layers = nn.ModuleList([nn.Linear(input_channels, 110)])  # Input layer to 1st hidden layer
        self.layers.append(nn.Linear(110, 10))  # Last hidden layer to output layer

    def forward(self, x):
        # Defining the activation function: leaky ReLU for hidden layers, softmax for output layer
        for i in range(len(self.layers)-1):  # leaky ReLU for all except the output layer
            x = F.leaky_relu(self.layers[i](x))
        x = F.softmax(self.layers[-1](x), dim=1)  # Softmax for the output layer
        return x

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    # convolutional layers
    self.conv1 = nn.Sequential(
      nn.Conv2d(
        in_channels = 1,
        out_channels = 32,
        kernel_size = 7
      ),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(
        in_channels = 32,
        out_channels = 16,
        kernel_size = 5
      ),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2),
    )

    # fully connected layers
    self.out = nn.Linear(16*3*3, output_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)  # flatten the output of conv2
    x = self.out(x)
    x = F.softmax(x, dim=1)
    return x