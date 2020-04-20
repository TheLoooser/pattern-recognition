# Adapted from https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
from random import seed, randint
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

# read in the MNIST data set
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

data_path = 'mnist-permutated-png-format/mnist/train'
train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

data_path = 'mnist-permutated-png-format/mnist/val'
val_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

data_path = 'mnist-permutated-png-format/mnist/test'
test_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

# # show images
# # imshow(torchvision.utils.make_grid(images))
# # print(images.shape[0])
# # # imshow(torchvision.utils.make_grid(images.shape[0]))
# # imshow(torchvision.utils.make_grid(images.view(-1, 784)))

# create network
input_size = 784
hidden_sizes = [20]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.Sigmoid(),
                      nn.Linear(hidden_sizes[0], output_size),
                      nn.Softmax(dim=1))
print(model)

criterion = nn.L1Loss()  # using MSE (mean squared error)

# check if CUDA is available (to increase performance)
cuda = False
if torch.cuda.is_available():
    cuda = False
device = torch.device("cuda:0" if cuda else "cpu")


# function which takes an image and a label
# returns 1 if the network guesses the correct number (0 o/w)
def prediction(imge, labl):
    imge = imge.view(-1, 784)
    if cuda:
        imge = imge.to(device)
    with torch.no_grad():
        probas = model(imge)

    p = probas.cpu() if cuda else probas
    probb = list(p.numpy()[0])
    pred = probb.index(max(probb))
    if labl == pred:
        return 1  # correct guess
    else:
        return 0


# training procedure
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)  # lr is the learning rate
time0 = time()
epochs = 2
epoch = [i for i in range(epochs)]
train_error = []
val_error = []
for e in range(epochs):
    train_err = 0
    for image, label in trainloader:
        # Flatten MNIST image into a 784 long vector
        # image2 = image.view(image.shape[0], -1)
        image2 = image.view(-1, 784)
        zeros = np.zeros(10)
        zeros.put(label.item(), 1)
        lbl = torch.from_numpy(zeros)
        if cuda:
            model.to(device)
            image2, lbl = image2.to(device), lbl.to(device)

        # Training pass
        optimizer.zero_grad()

        output = model(image2)

        loss = criterion(output[0], lbl)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        # epoch_loss += loss.item()
        train_err += prediction(image, label)
    else:
        print("Epoch {} - Error: {}".format(e, train_err / len(trainloader)))
    train_error.append(1 - (train_err / len(trainloader)))

    # validation procedure
    val_err = 0
    for image, label in validationloader:
        val_err += prediction(image, label)
    val_error.append(1-(val_err / len(validationloader)))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)


# testing procedure (get accuracy of trained network)
correct_count, all_count = 0, 0
for idx, (image, label) in enumerate(testloader):
    img = image.view(-1, 784) # TODO change image to grayscale instead
    if cuda:
        img = img.to(device)
    with torch.no_grad():
        probs = model(img)

    ps = probs.cpu() if cuda else probs
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = label
    if true_label == pred_label:
        correct_count += 1
    all_count += 1

    # verify a few samples manually (to check if the network actually worked correctly)
    seed(1)
    rnd = randint(-10, 10)
    if idx % (1000 + rnd) == 0:
        # plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')
        # plt.show()
        imshow(torchvision.utils.make_grid(image))
        print("Predicted {} - Ground truth: {}".format(pred_label, true_label[0]))

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))

fig = plt.figure()
plt.plot(epoch, train_error, color='blue')
plt.scatter(epoch, val_error, color='red')
plt.legend(['Train Error', 'Test Error'], loc='upper right')
plt.xlabel('epochs')
plt.ylabel('error')
fig.show()
