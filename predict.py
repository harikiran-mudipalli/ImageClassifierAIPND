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
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Getting inputs from user
def get_input_args():
    parse = argparse.ArgumentParser(description='Predict given flower class using trained model')
    parse.add_argument('--img', type=str, default='flowers/test/1/image_06760.jpg', help='Select the image to be classified')
    parse.add_argument('--check_point', type=str, default='checkpoint.pth', help='Select the check point to be loaded')
    parse.add_argument('--gpu', type=bool, default=False, help='Select to use GPU or run without GPU')
    parse.add_argument('--topk', type=int, default=5, help='Print top K classes with their probabilities')
    parse.add_argument('--category_to_name', type=str, default='cat_to_name.json', help='Load the JSON file that maps the class values to category names')
    
    return parse.parse_args()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True) if checkpoint['arch'] == 'VGG' else models.densenet161(pretrained=True)
    model.to(device)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im=Image.open(image)
    width=im.size[0]
    height=im.size[1]
    AspectRatio=width/height
    
    if width <= height:
        im=im.resize((256,int(256/AspectRatio)))
    else:
        im=im.resize((int(256*AspectRatio),256))
    
    midWidth=im.size[0]/2
    midHeight=im.size[1]/2
    cropped_im=im.crop((midWidth-112, midHeight-112, midWidth+112, midHeight+112))
    
    np_image=np.asarray(cropped_im)/255
    means=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    normalized_image=(np_image-means)/std
    final_image=normalized_image.transpose((2, 0, 1))
    
    return torch.from_numpy(final_image)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze(0).float()
    
    device = torch.device('cuda:0' if args.gpu else 'cpu')
    
    image = image.to(device)
    
    model = load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
    
    probs, indices = torch.topk(ps, topk)
    Probs = np.array(probs.data[0])
    Indices = np.array(indices.data[0])
    
    # A dictionary mapping the integer encoded categories to the actual names of the flowers
    with open(args.category_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    # invert class_to_idx
    idx_to_class = {idx:Class for Class,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in Indices]
    labels = [cat_to_name[Class] for Class in classes]

    return Probs,labels

# Print top K class with their probabilities
args = get_input_args()
probs, classes = predict(args.img, args.check_point, args.topk)
print('Type     Probability\n')
for prob, Class in zip(probs, classes):
    print('%s: %f' % (Class, prob))