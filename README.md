# LeNet-Inspired CNN for MNIST Classification

A clean PyTorch implementation of a **LeNet-inspired Convolutional Neural Network (CNN)** for handwritten digit classification on the **MNIST** dataset.

> **Note:** This is not an exact reproduction of the original LeNet-5 architecture. It's a lightly modified, LeNet-inspired CNN adapted for 28×28 MNIST images.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Key Takeaways](#key-takeaways)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview

This project implements a compact CNN that combines convolutional feature extraction with a fully connected classifier. The model takes 28×28 grayscale MNIST images as input and predicts one of ten digit classes (0–9).

The complete pipeline includes:

- MNIST dataset loading and preprocessing
- Image-to-tensor conversion
- CNN-based feature extraction
- Fully connected classification head
- Cross-entropy loss
- Adam optimization
- Automatic GPU/CPU device selection
- Per-epoch training and test-set evaluation

---

## Model Architecture

The network consists of two convolutional blocks (conv → ReLU → average pool) followed by a three-layer fully connected classifier.

```text
Input: 1 × 28 × 28
        │
        ▼
Conv2D (1 → 6, 5×5)      →  6 × 24 × 24
        │
       ReLU
        │
  AvgPool (2×2, stride 2) →  6 × 12 × 12
        │
        ▼
Conv2D (6 → 16, 5×5)     → 16 × 8 × 8
        │
       ReLU
        │
  AvgPool (2×2, stride 2) → 16 × 4 × 4
        │
        ▼
     Flatten              →  256
        │
        ▼
   Linear (256 → 120)     →  120
        │
       ReLU
        │
   Linear (120 → 84)      →   84
        │
       ReLU
        │
   Linear (84 → 10)       →   10
        │
        ▼
    Digit Class (0–9)
```

### Architecture Summary

| Layer   | Configuration       | Output Shape  |
|---------|---------------------|---------------|
| Input   | Grayscale image     | 1 × 28 × 28   |
| Conv2D  | 1 → 6, 5×5          | 6 × 24 × 24   |
| ReLU    | —                   | 6 × 24 × 24   |
| AvgPool | 2×2, stride 2       | 6 × 12 × 12   |
| Conv2D  | 6 → 16, 5×5         | 16 × 8 × 8    |
| ReLU    | —                   | 16 × 8 × 8    |
| AvgPool | 2×2, stride 2       | 16 × 4 × 4    |
| Flatten | —                   | 256           |
| Linear  | 256 → 120           | 120           |
| ReLU    | —                   | 120           |
| Linear  | 120 → 84            | 84            |
| ReLU    | —                   | 84            |
| Linear  | 84 → 10             | 10            |

---

## Dataset

The project uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset:

| Property         | Value              |
|-------------------|--------------------|
| Training samples | 60,000             |
| Test samples      | 10,000             |
| Image size        | 28 × 28 pixels     |
| Channels          | 1 (grayscale)      |
| Classes           | 10 (digits 0–9)    |

Images are converted to tensors using `torchvision.transforms.ToTensor()`.

---

## Training Configuration

| Parameter          | Value              |
|----------------------|--------------------|
| Batch size          | 128                |
| Learning rate        | 0.001              |
| Optimizer            | Adam               |
| Loss function         | Cross-Entropy Loss |
| Epochs                | 5                   |
| Input resolution      | 28 × 28             |
| Number of classes     | 10                  |

The implementation automatically uses a CUDA-enabled GPU when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Training Loop

Each epoch performs:

1. Forward propagation
2. Cross-entropy loss computation
3. Backpropagation
4. Adam parameter update
5. Training loss and accuracy tracking

The model is evaluated on the MNIST test set after every epoch, tracking:

- Training loss
- Training accuracy
- Test loss
- Test accuracy

Example output:

```text
Epoch 1/5, Train Loss: ..., Train Acc: ..., Test Loss: ..., Test Acc: ...
Epoch 2/5, Train Loss: ..., Train Acc: ..., Test Loss: ..., Test Acc: ...
...
Epoch 5/5, Train Loss: ..., Train Acc: ..., Test Loss: ..., Test Acc: ...
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ratul-dutta/lenet-inspired-cnn-mnist.git
cd lenet-inspired-cnn-mnist
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install torch torchvision numpy
```

---

## Usage

Run the training script:

```bash
python lenet_mnist.py
```

The MNIST dataset is automatically downloaded to the `./data` directory on first run if not already present.

---

## Results

| Metric          | Result       |
|-------------------|--------------|
| Test Accuracy     | _Add final result_ |
| Epochs             | 5            |
| Batch Size         | 128          |
| Optimizer          | Adam         |
| Learning Rate      | 1e-3         |

> Update this table with your final training run results.

---

## Project Structure

```text
lenet-inspired-cnn-mnist/
│
├── README.md
├── lenet_mnist.py
└── requirements.txt
```

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- CUDA (when available)

---

## Key Takeaways

This project demonstrates the fundamental workflow of a convolutional neural network for image classification, including:

- Convolution-based feature extraction
- Spatial downsampling using average pooling
- Fully connected classification
- Backpropagation and gradient-based optimization
- Model evaluation on unseen data

---

## Future Improvements

- [ ] Replace average pooling with max pooling and compare performance
- [ ] Add batch normalization between conv layers
- [ ] Experiment with dropout for regularization
- [ ] Add a confusion matrix and per-class accuracy breakdown
- [ ] Log training curves with TensorBoard or Weights & Biases
- [ ] Add a saved-checkpoint / inference script

---

## Author

**Ratul Dutta**
M.Sc. Mathematics & Computing, Banaras Hindu University
[GitHub](https://github.com/ratul-dutta) · [LinkedIn](https://www.linkedin.com/in/ratul-dutta-33441b317)

---

## License

This project is available under the [MIT License](LICENSE).
