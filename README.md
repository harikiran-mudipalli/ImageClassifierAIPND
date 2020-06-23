# ImageClassifierAIPND
 Capstone Project from Udacity's AI Programming with Python Nano Degree</br>
Python application that can train an Image Classifier on a dataset, then predict new Images using Trained model.

This Application uses PyTorch framework.

## Training
To train a model, run `train.py`, with desired arguments.
### Example command
```
python train.py --arch densenet --hidden_units 1000,500 --epochs 5 --lr 0.003 --gpu true --data_directory flowers/ --save_directory checkpoint.pth
```
### Usage
```
train.py [-h] [--data_directory DATA_DIR] [--save_directory SAVE_DIR] [--arch ARCH]
                [--lr LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu GPU]
                
arguments:
  -h, --help                  show this help message and exit
  --data_directory DATA_DIR   path to image folder
  --save_directory SAVE_DIR   folder where model checkpoints gets saved to
  --gpu GPU                   whether gpu should be used for or not
  --arch ARCH                 choose between vgg and densenet
  --lr LEARNING_RATE          learning_rate for model
  --hidden_units HIDDEN_UNITS hidden_units for model
  --epochs EPOCHS             epochs for model
```
After training the model, {checkpoint}.pth file is observed in your working directory which contains the saved parameters of trained model and using those parameters prediction is done.

## Prediction
To make prediction, run `predict.py`, with desired arguments.
### Example command
```
python predict.py --check_point checkpoint.pth --img flowers/test/32/image_051001.jpg --top_k 5 --category_to_names cat_to_name.json --gpu true
```
### Usage
```
usage: predict.py [-h] [--img INPUT] [--check_point CHECKPOINT]
                  [--topk TOP_K] [--category_to_name CATEGORY_NAMES]
                  [--gpu GPU]

Provide input, checkpoint, top_k, category_names and gpu

optional arguments:
  -h, --help                        show this help message and exit
  --img INPUT                       path to input image
  --check_point CHECKPOINT          path to checkpoint
  --gpu GPU                         whether gpu should be used for or not
  --topk TOP_K                      number of top_k to show
  --category_to_name CATEGORY_NAMES path to cat names file
```
The commands above will use default values for all other unspecified arguments.
After prediction is complete, the application outputs top k classes along with their predicted probabilities.
