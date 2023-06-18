# Language-Classification-CNN
## Overview
#### This repository contains code for a Convolutional Neural Network (CNN) designed to perform language classification. The CNN is trained on a dataset of text samples from 6 languages and can predict the language of a given audio entry.
#### Language classification is a task in natural language processing (NLP) that involves determining the language of a given piece of audio. This CNN-based language classifier utilizes the power of convolutional layers to automatically learn relevant features from the input clips and make accurate language predictions. The model is trained on a diverse dataset of audio samples from various languages, allowing it to generalize well to unseen data.
## Dataset
#### The dataset consists of individual waveforms. These contain the shape of the sound signal over time. Because our audio clips are digital, the waveforms consist of sound amplitude at individual timesteps. The clips here are sampled at 8 kilohertz (kHz), meaning there are 8000 amplitude measurements for every second. Each audio clip is 5 seconds long and has 8000 · 5 = 40000 measurements in total.
#### Dataset: https://drive.google.com/drive/folders/1KpYCBHecXojn2voG9gRB1iRC7nKMSEUZ?usp=sharing
## Model architecture
#### The CNN architecture employed for language classification consists of several layers designed to extract important features from the input. The following is the architecture of the model:
#### 1. Normalization Layer: apply normalization to the input data
#### 2. Transformation Layer: transform the normalized data into mel spectrogram.
#### 3. Convolutional Layers: convolutional filters to capture local patterns and features.
#### 4. MaxPooling Layer: Extract the most important features from the convolutional layers.
#### 5. Fully Connected Layers: Perform classification based on the extracted features.
#### 6. Output Layer: Generates language predictions using softmax activation.
