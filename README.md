# Flower-Image-Classification-Using-Pytorch-and-CNN

This repository contains a Jupyter Notebook that demonstrates the process of classifying different types of flowers using Convolutional Neural Networks (CNNs). The project involves data preprocessing, model training, and evaluation using a dataset of flower images.

# Project Overview
The goal of this project is to build a deep learning model that can accurately classify images of flowers into different categories. The primary algorithm used in this project is a Convolutional Neural Network (CNN), which is well-suited for image classification tasks.

# Dataset
The dataset used in this project consists of images of flowers belonging to different classes. The classes include:

# Daisy
# Dandelion
Each class contains multiple images that are used for training and validation. The dataset is divided into training and validation sets.

# Steps Followed
# 1. Importing Libraries
The first step involves importing the necessary libraries for data processing, model building, and evaluation. Key libraries include:

torch for building and training the neural network

torchvision for handling image datasets and transformations

os for file handling

# 2. Data Preprocessing
Data Augmentation and Normalization: The images are transformed using data augmentation techniques to enhance the model's generalization ability. The images are also normalized to have a similar scale.

Data Loading: The transformed images are loaded into PyTorch DataLoaders, which are used to feed data into the model during training and validation.

# 3. Model Architecture
The Convolutional Neural Network (CNN) is built using a pre-trained model (e.g., ResNet) from the torchvision library. The model is fine-tuned by modifying the final layers to match the number of classes in the dataset.

# 4. Training the Model
The model is trained using the training dataset. The loss function used is Cross-Entropy Loss, and the optimizer used is Stochastic Gradient Descent (SGD).

# 5. Model Evaluation
The model is evaluated on the validation dataset. The accuracy, loss, and other metrics are recorded to assess the performance of the model.

# 6. Saving the Model
The trained model is saved for future use, which allows for quick deployment without needing to retrain the model.

# 7. Conclusion
The CNN model provides an effective method for classifying flower images into different categories. The use of transfer learning with a pre-trained model significantly improves the accuracy and reduces the training time.

# Dependencies
To run the notebook, ensure you have the following dependencies installed:
pip install torch torchvision
