# Importing required Libraries
import numpy as np
import pandas as pd
import time
import json
import torch
import argparse

from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Options for user : Data directory, with GPU or without GPU, and model architecture
parser = argparse.ArgumentParser(description='Train a model to classify flowers')
parser.add_argument('--data_directory', type=str, default='flowers', help='Directory of Training and Testing Images')
parser.add_argument('--save_directory', type=str, default='checkpoint.pth', help='Directory containing saved checkpoints')
parser.add_argument('--gpu', type=bool, default=False, help='Choose between GPU or non GPU mode for training')
parser.add_argument('--arch', type=str, default='VGG', help='Choose VGG or Densenet architecture')
parser.add_argument('--lr', type=float, default=0.0003, help='Choose learning rate for the model')
parser.add_argument('--hidden_units', type=int, default=500, help='Choose the number of hidden units for the model network')
parser.add_argument('--epochs', type=int, default=3, help='Choose training epochs for the model')

# Store user choices in a variable
args = parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for train, validation, and test sets
transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

transforms_validation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

transforms_testing = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder 
train_data = datasets.ImageFolder(train_dir, transform=transforms_training)
validate_data = datasets.ImageFolder(valid_dir, transform=transforms_validation)
test_data = datasets.ImageFolder(test_dir, transform=transforms_testing)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validate_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Build the model with choosen architecture
if args.arch == 'VGG':
    model = models.vgg16(pretrained=True)
    # Number of input nodes
    num_features = model.classifier[0].in_features
else:
    model = models.densenet161(pretrained=True)
    num_features = model.classifier[0].in_features

# Freeze parameters to avoid backprop through them
for param in model.parameters():
    param.requires_grad = False
    
# Building Classifier
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(num_features, 2000)),
                           ('relu', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=0.4)),
                           ('fc2', nn.Linear(2000, args.hidden_units)),
                           ('relu', nn.ReLU()),
                           ('dropout2', nn.Dropout(p=0.3)),
                           ('fc3', nn.Linear(args.hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                            ]))

model.classifier = classifier

# define optimizer and criterion
criterion = nn.NLLLoss()
    # Train only the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

# Train the network
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 40

device = torch.device('cuda:0' if args.gpu else 'cpu')

model.to(device)

start = time.time()
for e in range(epochs):
    # Model in training mode, dropout is on
    model.train()
    for inputs, labels in iter(trainloader):
        steps += 1
            
        # Wrap images and labels in Variables so we can calculate gradients
        inputs = Variable(inputs)
        targets = Variable(labels)
            
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
            
        optimizer.zero_grad()
            
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
            
        running_loss += loss.data[0]
            
        if steps % print_every == 0:
            # Model in inference mode, dropout is off
            model.eval()
                
            accuracy = 0
            test_loss = 0
            for ii, (inputs, labels) in enumerate(validloader):

                # Set volatile to True so we don't save the history
                inputs = Variable(inputs, volatile=True)
                labels = Variable(labels, volatile=True)
                 
                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(device), labels.to(device)

                output = model.forward(inputs)
                test_loss += criterion(output, labels).data[0]

                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Valid Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                    "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0

            # Make sure dropout is on for training
            model.train()
time_elapsed = time.time() - start
print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Do validation on the test set
model.eval()

correct = 0
total = 0
with torch.no_grad():
    start = time.time()
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
time_elapsed = time.time() - start
print("Accuracy on test data:",(100*correct)/total,'%')
print('Testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch':args.arch,
              'input': num_features,
              'output':102,
              'epochs':args.epochs,
              'learning_rate':args.lr,
              'dropout1':0.4,
              'dropout2':0.3,
              'batch_size':32,
              'classifier':classifier,
              'state_dict':model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, args.save_directory)