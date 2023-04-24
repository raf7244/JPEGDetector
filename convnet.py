import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import io

from torchvision.datasets import ImageFolder


# device config
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 500
batch_size = 100
learning_rate = 0.02


data_train = ImageFolder(root='C:/python/masters/train_dir', transform=transforms.ToTensor())

data_test = ImageFolder(root='C:/python/masters/test_dir', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)


classes = ('2', '4', '6', '8', '10')

def imshow(img):
    #img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()


# implement conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(20, 100, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(100*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten
        x = x.view(-1, 100*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# testing
def test():
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(5)]
        n_class_samples = [0 for i in range(5)]
        #print(n_class_correct)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples+=labels.shape[0]
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if(label == pred):
                    n_class_correct[label] +=1
                n_class_samples[label] +=1

        acc = n_correct/n_samples * 100.0
        print(f'{acc:.4f}', end=' ')

        for i in range(5):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'{acc:.4f}', end=' ')
        print()

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass

        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'{epoch+1} {loss.item():.4f}', end=' ')
    test()
    #print()

print('Finished training')

