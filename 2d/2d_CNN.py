import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, freeze_support

# Set this to true if you want to use cuda if possible
USE_CUDA = False

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28, 28)

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels ands the kernel size
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=7, stride=3),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(1536, 10)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.fc(x)
        return x

def run():
    freeze_support()
    # read in the MNIST data set, convert to grayscale as it is read in as 3 channel
    transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    data_path = 'mnist-permutated-png-format/mnist/train'
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    data_path = 'mnist-permutated-png-format/mnist/val'
    val_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    data_path = 'mnist-permutated-png-format/mnist/test'
    test_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    net = PR_CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # CUDA setup
    cuda = False
    if USE_CUDA:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            cuda = True
            net.to(device)

    epochs = 10
    train_error = []
    val_error = []

    log_interval = 10000
    # run the main training loop
    for epoch in range(epochs):
        correct = 0
        size = len(train_loader.dataset)
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # inputs, labels = data
            net_out = net(data)
            loss = criterion(net_out, target)
            # backpropagation
            loss.backward()
            # optimize weights
            optimizer.step()

            # get the index of the max log-probability
            pred = net_out.data.max(1)[1]
            # compares label and prediction tensors, returns 1 if same, 0 otherwise
            correct += int(pred.eq(target.data).sum())
        
            # logging for supervision
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss.data.item()))

        # store train error
        train_error.append(100. * correct / size)

        # validate model after each epoch
        validation_correct = 0
        for data, target in validation_loader:
            if cuda:
                data, target = data.to(device), target.to(device)
            net_out = net(data)
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            # cast from tensor to int
            validation_correct += int(pred.eq(target.data).sum())
        # store validation_error
        val_error.append(100. * validation_correct / len(validation_loader.dataset))

    # run a test loop
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.to(device), target.to(device)
            net_out = net(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # plot results
    fig = plt.figure()
    plt.plot(train_error, color='blue')
    plt.plot(val_error, color='red')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # for older plt versions
    # fig.show()
    plt.show()

if __name__ == '__main__':
    run()
