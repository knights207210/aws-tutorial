import pickle
import gzip
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

PATH = '/home/ec2-user/data/'
FILENAME = '25635053'

def get_files(path,filename):
    with gzip.open((PATH + FILENAME), "rb") as file:
        ((x_train, y_train), (x_val, y_val), _) = pickle.load(file, encoding='latin-1')
    return x_train, y_train, x_val, y_val

def tensor_map(x_train,y_train,x_val,y_val): return map(torch.tensor,(x_train,y_train,x_val,y_val))

def preprocess(x):
    return x.view(-1, 1, 28, 28)

def conv(in_size, out_size, pad=1):
    return nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=pad)

class ResBlock(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int, pad:int):
        super().__init__()
        self.conv1 = conv(in_size, hidden_size, pad)
        self.conv2 = conv(hidden_size, out_size, pad)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x): return x + self.convblock(x) # skip connection

class ResNet(nn.Module):
    
    def __init__(self, n_classes=10):
        super().__init__()
        self.res1 = ResBlock(1, 8, 16, 15)
        self.res2 = ResBlock(16, 32, 16, 15)
        self.conv = conv(16, n_classes)
        self.batchnorm = nn.BatchNorm2d(n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        x = preprocess(x)
        x = self.res1(x)
        x = self.res2(x) 
        x = self.maxpool(self.batchnorm(self.conv(x)))
        return x.view(x.size(0), -1)

def loss_batch(model, loss_func, xb, yb, opt=None, scheduler=None):
    loss = loss_func(model(xb), yb)
    acc = accuracy(model(xb), yb)
    if opt is not None:
        loss.backward()
        if scheduler is not None:
            scheduler.step()
        opt.step()
        opt.zero_grad()
    return acc, loss.item(), len(xb)

def accuracy(out, yb):
    # in PyTorch one cannot take the mean of ints
    # thus, values have to be converted into floats first
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def get_model():
    model = ResNet()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, optimizer

def get_data_batches(x_train, y_train, x_val, y_val, bs):
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    # DataLoader = generator
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs * 2),
    )

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, scheduler=None):
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        # iterate over data loader object (generator)
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt, scheduler)

        print("finish train epoch: ", epoch)
        print("start eval")

        model.eval()
        # no gradient computation for evaluation mode
        with torch.no_grad():
            accs, losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        
        #NOTE: important to multiply with batch size and sum over values 
        #      to account for varying batch sizes
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)

        print("Epoch:", epoch+1)
        print("Loss: ", val_loss)
        print("Accuracy: ", val_acc)
        print()

bs=64 #128
lr=0.01
n_epochs = 5
loss_func = F.cross_entropy

# get data set
x_train, y_train, x_val, y_val = get_files(PATH, FILENAME)

print("dataset loading succeed")

# map tensor function to all inputs (X) and targets (Y) to create tensor data sets
x_train, y_train, x_val, y_val = tensor_map(x_train, y_train, x_val, y_val)

print("map tensor function")

# get math.ceil(x_train.shape[0]/batch size) train and val mini batches of size bs
train_dl, val_dl = get_data_batches(x_train, y_train, x_val, y_val, bs)

print("preparing data format succeed")

# get model and optimizer
model, opt = get_model()

# train
print("start training...")
fit(n_epochs, model, loss_func, opt, train_dl, val_dl)


