# Music Genre Classification

A deep learning project for music genre classification using the GTZAN dataset with PyTorch.

## Project Overview

This project implements two different neural network architectures to classify music into different genres:
- AlexNet (in recommend.py)
- ResNet (in recommend_resnet.py)

## Features

- Audio preprocessing and feature extraction
- Data augmentation for audio samples
- Training and evaluation pipeline
- TensorBoard integration for monitoring training progress
- Hyperparameter management system

## Requirements

- PyTorch
- torchaudio
- pandas
- matplotlib
- torchsummary
- tensorboard

## Dataset

The project uses the GTZAN dataset which contains 1000 audio tracks, each 30 seconds long, with 10 genres (100 tracks per genre):
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Implementation Details

### Data Processing

- Audio samples are standardized to a consistent sample rate
- Mel spectrogram transformation is applied to convert audio signals to image-like data
- Data augmentation techniques are applied to improve model generalization

### Model Architectures

1. **AlexNet**
   - Adapted for audio classification
   - Uses convolutional layers followed by fully connected layers

2. **ResNet**
   - Implements residual connections to allow for deeper network training
   - Available in both ResNet-18 and ResNet-34 configurations

### Training

The training pipeline includes:
- Custom dataset class for audio data loading
- Hyperparameter management
- Training progress tracking
- Model evaluation on test data

## Usage

To train the AlexNet model:
```
python recommend.py
```

To train the ResNet model:
```
python recommend_resnet.py
```

## Performance

Both models are evaluated based on:
- Classification accuracy
- Loss convergence
- Training and inference speed

The training progress can be monitored using TensorBoard.
