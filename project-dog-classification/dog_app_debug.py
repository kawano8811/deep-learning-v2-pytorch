import numpy as np
import os
from torchvision import datasets
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()

mean = [0.48640698, 0.45601872, 0.39183483]
std = [0.23047441, 0.2256413, 0.22351243]
# normalize
normalize = transforms.Normalize(mean=mean, std=std)

img_path = '/work/notebook_work/study/udacity/DeepLearning/deep-learning-v2-pytorch/project-dog-classification'

# train data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=10),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    normalize,
])

train_data = datasets.ImageFolder(img_path + '/dogImages/train/', transform=transform_train)

# validation data
transform_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

valid_data = datasets.ImageFolder(img_path + '/dogImages/valid/', transform=transform_valid)

# test data
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

test_data = datasets.ImageFolder(img_path + '/dogImages/test/', transform=transform_test)


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (3x224x224 -> 32x224x224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # pooling layer (32x224x224 -> 32x112x112)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # convolutional layer (32x112x112 -> 64x112x112)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # pooling layer (64x112x112 -> 64x56x56)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # convolutional layer (64x56x56 -> 128x56x56)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # pooling layer (128x56x56 -> 128x28x28)
        self.max6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # convolutional layer (128x28x28 -> 256x28x28)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # max pooling layer (256x28x28 -> 256x14x14)
        self.max8 = nn.MaxPool2d(kernel_size=2, stride=2)
        # linear layer (256x14x14 -> 4096)
        self.fc9 = nn.Linear(256 * 14 * 14, 4096)
        # linear layer (4096 -> 133)
        self.fc10 = nn.Linear(4096, 133)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.max2(x)
        x = F.relu(self.conv3(x))
        x = self.max4(x)
        x = F.relu(self.conv5(x))
        x = self.max6(x)
        x = F.relu(self.conv7(x))
        x = self.max8(x)
        x = x.view(-1, 256 * 14 * 14) # flatten image input
        x = self.dropout(x)
        x = F.relu(self.fc9(x))
        x = self.dropout(x)
        x = self.fc10(x)
        return x
    

    # ### TODO: choose an architecture, and complete the class
    # def __init__(self):
    #     super(Net, self).__init__()
    #     ## Define layers of a CNN
    #     # Convolution1: 3 * 224 * 224 -> 96 * 55 * 55
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    #     # Max Pooling2: 96 * 55 * 55 -> 96 * 27 * 27
    #     self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)
    #     # Convolutoin3: 96 * 27 * 27 -> 256 * 27 * 27
    #     self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
    #     # Max Pooling: 256 * 27 * 27 -> 256 * 13 * 13
    #     self.max4 = nn.MaxPool2d(kernel_size=3, stride=2)
    #     # Convolution: 256 * 13 * 13 -> 384 * 13 * 13
    #     self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
    #     # Convolution: 384 * 13 * 13 -> 384 * 13 * 13
    #     self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
    #     # Convolution: 384 * 13 * 13 -> 256 * 13 * 13
    #     self.conv7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    #     # Max Pooling: 256 * 13 * 13 -> 256 * 6 * 6
    #     self.max8 = nn.MaxPool2d(kernel_size=3, stride=2)
    #     # Linear: 256 * 6 * 6 -> 4096
    #     self.linear9 = nn.Linear(256 * 6 * 6, 4096)
    #     # Linear: 4096 -> 4096
    #     self.linear10 = nn.Linear(4096, 4096)
    #     # Linear: 4096 -> 133
    #     self.linear11 = nn.Linear(4096, 133)
    #     # dropout
    #     self.dropout = nn.Dropout()
        
    # def forward(self, x):
    #     ## Define forward behavior
    #     x = F.relu(self.conv1(x))
    #     x = self.max2(x)
    #     x = F.relu(self.conv3(x))
    #     x = self.max4(x)
    #     x = F.relu(self.conv5(x))
    #     x = F.relu(self.conv6(x))
    #     x = F.relu(self.conv7(x))
    #     x = self.max8(x)
    #     # x = self.dropout(x)
    #     x = x.view(-1, 256 * 6 * 6) # flatten
    #     x = F.relu(self.linear9(x))
    #     x = self.dropout(x)
    #     x = F.relu(self.linear10(x))
    #     x = self.linear11(x)
    #     return x
    
#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)

# batch size
batch_size = 32
# number of subprocesses to use for data loading
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

loaders_scratch = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}



# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            # update training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            print('{:.6f}'.format(train_loss))

            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update the average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # # calculate average losses
        # train_loss = train_loss / len(train_loader)
        # valid_loss = valid_loss / len(valid_loader)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# train the model
model_scratch = train(50, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))

