# 🧬 Generative Adversarial Networks (GAN) on MedMNIST - PathMNIST

This project demonstrates how to implement and train a **Generative Adversarial Network (GAN)** using **PyTorch** to generate synthetic medical images from the **PathMNIST** dataset. It’s designed to help **beginners** understand how GANs work step-by-step.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Evaluation (FID Score)](#-evaluation-fid-score)
- [TensorBoard Logging](#-tensorboard-logging)
- [Results](#-results)
- [Author](#-author)

---

## 📌 Overview

A **GAN** consists of two neural networks:
- **Generator**: Learns to create images from random noise.
- **Discriminator**: Learns to distinguish between real and generated images.

They train together in a game where the generator tries to fool the discriminator, and the discriminator tries not to be fooled.

---

## ⚙️ Installation

Install the required Python packages:

```bash
pip install torch torchvision medmnist tensorboard scipy
```

---

## 🧬 Dataset

We use the **PathMNIST** dataset which consists of 28x28 RGB images of 9 tissue types from pathology slides.

### 📥 Loading the Dataset

```python
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = PathMNIST(root="./data", split="train", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## 🧠 Model Architecture

### 🧪 Generator

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)
```

### 🛡️ Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)
```

---

## 🏋️ Training Process

Basic training loop:

```python
for epoch in range(num_epochs):
    for imgs, _ in dataloader:
        # 1. Train Discriminator on real and fake images
        # 2. Train Generator to fool the Discriminator
        # 3. Log losses and generated samples
```

Each epoch improves the generator’s ability to create realistic images.

---

## 📊 Evaluation (FID Score)

**Fréchet Inception Distance (FID)** compares the distribution of generated images with real ones.

```python
from scipy.linalg import sqrtm
import numpy as np

def compute_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1 @ sigma2)
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean.real)
    return fid
```

Lower FID = More realistic generated images.

---

## 📈 TensorBoard Logging

Monitor training with TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/GAN_MedMNIST")

# During training
writer.add_scalar("Loss/Generator", g_loss, epoch)
writer.add_scalar("Loss/Discriminator", d_loss, epoch)
writer.add_images("Generated Images", fake_images, epoch)
```

Launch TensorBoard in terminal:

```bash
tensorboard --logdir=runs
```

---

## 🖼️ Results

- Early outputs are noisy.
- Over time, generated images resemble real histopathology slides.
- TensorBoard helps visualize learning progress.

---

## 👨‍💻 Author

**Vaibhav Uniyal**  
B.Tech in Artificial Intelligence and Machine Learning  
Symbiosis Institute of Technology, Pune  

Feel free to ⭐ this project if it helped you understand GANs!

---
