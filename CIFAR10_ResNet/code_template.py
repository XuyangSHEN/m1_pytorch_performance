import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Device Configure
device = torch.device('mps') # m1 GPU
# device = torch.device('cpu') # CPU
# device = torch.device('cuda') # nv GPU


batch_size = 512
num_workers = 8

# Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)


# Model
net = resnet50(pretrained=False).to(device)

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train_engine(epochs=10):
    net.train()
    
    total_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        correct, total = 0, 0
        running_loss = 0.0
        start_time = time.time()
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}: takes {(time.time() - start_time):.2f} seconds, loss: {(running_loss / (i+1)):.3f}, acc: {(correct/total*100):.2f}%')
    
    total_time = time.time() - total_time
    print(f'Finished training in {total_time:.2f} seconds; Avg: {(total_time/epochs):.2f} seconds')
    
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Test accuracy:{(correct/total*100):.2f}%')


if __name__ == '__main__':
    # train
    train_engine(epochs = 10)
