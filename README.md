Image Recognition:

In Image recognition, we input an image into a neural network and get a label (that belongs to a pre-defined class) for that image as an output. There can be multiple classes for the labeled image. If it belongs to a single class, then we call it recognition; if there are multiple classes, we call it classification. 

In this article, we use a flower dataset with 3670  images with five classes labeled as daisy, dandelion, roses, sunflowers, and tulips. The Image Classification model consists of the following steps:

Understand data and load data: In this stage, we need to collect image data and label them. If the images are downloaded from other sources, then also they must be preprocessed before using them for training. 
Build input pipeline: Tensorflow APIs allow us to create input pipelines to generate input data and preprocess them effectively for the training process. The pipeline for an image model aggregates data from files in a distributed file system applies random perturbations to each image and merges randomly selected images into a batch for training.
Build the model: In this stage, we make choices about parameters and hyperparameters and make decisions about the number of layers to be used in our model. We decide on the input and output sizes of the layers, along with the activation function.
Train the model: After creating a model, we must create an instance of the model and fit it with our training data. 
Test the model: The crucial part of this stage is to estimate: the amount of time the model takes to train and specify the length of training for a network depending on the number of epochs to train over.
Evaluate and improve the accuracy: Evaluating a model means comparing its performance against a validation dataset to analyze its performance through different metrics. The most common metric is ‘Accuracy’, calculated by dividing the amount of correctly classified images by the total number of images in our dataset.

---
Step 1 : Importing TensorFlow and other libraries 
The first step is to import the necessary libraries and modules such as pyplot, NumPy library, tensor-TensorFlow, os, and PIL.


# Importing libraries 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
  
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential  

---
