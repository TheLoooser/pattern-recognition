"""
CNN with 3 conv layers and a fully connected classification layer
PATTERN RECOGNITION EXERCISE:
Fix the three lines below marked with PR_FILL_HERE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, freeze_support

USE_CUDA = True


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


PATH = './CNN_net.pth'


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def run():
    freeze_support()
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int((5 / 6) * len(train_set))
    validation_size = len(train_set) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_set, [train_size, validation_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # ===========================
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

    epochs = 60
    log_interval = 10000
    size = len(train_loader.dataset)
    train_error = []
    train_loss = []
    val_error = []
    val_loss = []

    # run the main training loop
    for epoch in range(epochs):
        correct = 0
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            # data = data.view(-1, 28 * 28)
            if cuda:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
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
            loss_sum += loss.data.item()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss.data.item()))

        # store train error and loss
        train_error.append(100. * correct / size)
        train_loss.append(loss_sum/len(train_loader.dataset))

        # validate model after each epoch
        validation_correct = 0
        loss_sum = 0
        for data, target in validation_loader:
            if cuda:
                data, target = data.to(device), target.to(device)
            net_out = net(data)
            loss_sum += criterion(net_out,target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            # cast from tensor to int
            validation_correct += int(pred.eq(target.data).sum())
            loss = criterion(net_out, target)

        # store validation_error and loss
        val_error.append(100. * validation_correct / len(validation_loader.dataset))
        val_loss.append(loss_sum/len(validation_loader.dataset))

    torch.save(net.state_dict(), PATH)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # images = images.view(-1, 28 * 28)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # net = PR_CNN()
    # net.load_state_dict(torch.load(PATH))
    if cuda:
        images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # run a test loop
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = Variable(data), Variable(target)
            # data = data.view(-1, 28 * 28)
            if cuda:
                data, target = data.to(device), target.to(device)
            net_out = net(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        print('\nValidation set: \nAverage loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            val_loss[-1], val_error[-1]))

    # plot results
    plt.figure(1)
    plt.plot(train_error, color='blue')
    plt.plot(val_error, color='red')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.figure(2)
    plt.plot(train_loss, color='blue')
    plt.plot(val_loss, color='red')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    run()
