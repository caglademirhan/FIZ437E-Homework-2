import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

train_data = dsets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='./Data', train=False, transform=transforms.ToTensor())

batch = 100
iters = 3000
epochs = int(iters / (len(train_data) / batch))

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch, shuffle=False)

class LRModel(nn.Module):
    
    
    def __init__(self, input_size, num_classes):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
        




input_dim = 28**2 ; output_dim = 10

model = LRModel(input_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


criterio = nn.CrossEntropyLoss()

L_R = 1e-3

optimizer_regu = torch.optim.SGD(model.parameters(), lr=L_R)

print("Standart SGD: ")

iter = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 28*28).requires_grad_().to(device)
        labels = labels.to(device)

        optimizer_regu.zero_grad()

        outputs = model(images)

        loss = criterio(outputs, labels)

        loss.backward()

        optimizer_regu.step()

        iter = iter + 1

        if iter % 500 == 0:         
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                
                images = images.view(-1, 28*28).to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total = total + labels.size(0)

                if torch.cuda.is_available():
                    correct = correct + (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct = correct + (predicted == labels).sum()

            accuracy = 100 * correct.item() / total

            
            print('Iter: {}. L: {}. Ac: {}'.format(iter, loss.item(), accuracy)) # info



print("SGD with weight decay : ")

optimizer_w_regu = torch.optim.SGD(model.parameters(), lr=L_R, weight_decay=1e-5)

iter = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 28*28).requires_grad_().to(device)
        labels = labels.to(device)

        optimizer_w_regu.zero_grad()

        outputs = model(images)

        loss = criterio(outputs, labels)

        loss.backward()

        optimizer_w_regu.step()

        iter = iter + 1

        if iter % 500 == 0:         
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                
                images = images.view(-1, 28*28).to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total = total + labels.size(0)

                if torch.cuda.is_available():
                    correct = correct + (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct = correct + (predicted == labels).sum()

            accuracy = 100 * correct.item() / total

            
            print('Iter: {}. L: {}. Ac: {}'.format(iter, loss.item(), accuracy)) # info


