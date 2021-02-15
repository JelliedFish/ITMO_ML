
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


trainloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=b_size_train, shuffle=True)

testloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



epochs = 3
b_size_train = 60
batch_size_test = 900
learning_rate = 0.01
log_interval = 11

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print(dev)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d10, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(400, 10)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.max_pool3d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = x.view(-1, 3*2*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    net = Net(input_size, hidden_size, num_classes)
    net = net.to(device)
    o = optim.Adadelta(net.parameters(), lr=learning_rate)
    crit_train = nn.CrossEntropyLoss()
    crit_test = nn.CrossEntropyLoss()

    def train(self,network, trainloader, epoch, crit_train, o):
            for epoch in range(epochs):
                network.eval()

             for i, (set_of_images, labels) in enumerate(trainloader):
                labels = Variable(labels,)
                o.zero_grad()
                outputs = net(images)
                loss = crit_train(outputs,
                                 labels)
                loss.backward()
                o.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                          % (epoch + 1, epochs, i + 1, len(test_loader) // b_size_train, loss.data[0]))
        def test(network ,testloader, target):
            network.eval()
            test_loss = 0
            correct = 0


            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(testloader.dataset),
                100. * correct / len(testloader.dataset)))

            total = 0
            for images, labels in testloader:
                images = Variable(images.view(-1, 28 * 28))
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_loss = 100*(correct/total)


        torch.manual_seed(1)

        device = torch.device("cpu")

        model = Net().to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        torch.save(model.state_dict(), "mnist_cnn.pt")


        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root='./mnist', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=batch_size_train, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root='./mnist', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=batch_size_test, shuffle=True)


def matrix(self, num_convs, in_channels, out_channels):
    net = []
    for i in range(num_convs):
        in_c = in_channels + i * out_channels
        net.append(conv_block(in_c, out_channels))
    self.net = nn.ModuleList(net)
    self.out_channels = in_channels + num_convs * out_channels


def forward(self, X):
    for blk in self.net:
        Y = blk(X)
        X = torch.cat((X, Y), dim=1)
    return X

for epoch in range(1, epochs + 1):
  train(network, trainloader, epoch, crit_train, o)
  test(network, testloader, crit_test)