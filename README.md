Overview
A GAN consists of two neural networks:

Generator: Learns to create images from random noise.

Discriminator: Learns to distinguish between real and generated images.

They train together in a game where the generator tries to fool the discriminator, and the discriminator tries not to be fooled.

⚙️ Installation
Install required packages using pip:

bash
Copy
Edit
pip install torch torchvision medmnist tensorboard scipy
🧬 Dataset
We use the PathMNIST dataset which consists of colored 28x28 images of 9 tissue types from pathology slides.

Loading the Dataset:
python
Copy
Edit
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = PathMNIST(root="/content/data", split="train", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
🧠 Model Architecture
Generator:
Takes a random noise vector and outputs a 28x28 grayscale image.

python
Copy
Edit
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
Discriminator:
Classifies images as real or fake.

python
Copy
Edit
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
🏋️ Training Process
Initialize models and optimizers

Train Discriminator to distinguish real and generated images.

Train Generator to fool the discriminator.

Repeat!

Basic training loop:

python
Copy
Edit
for epoch in range(num_epochs):
    for imgs, _ in dataloader:
        # Train Discriminator
        # Train Generator
        # Log losses and generated samples
📊 Evaluation (FID Score)
We use Fréchet Inception Distance (FID) to evaluate how close the generated images are to the real ones.

python
Copy
Edit
def compute_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1 @ sigma2)
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2*covmean.real)
    return fid
📈 TensorBoard Logging
Track training progress visually:

python
Copy
Edit
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/GAN_MedMNIST")

# During training
writer.add_scalar("Loss/Generator", g_loss, epoch)
writer.add_scalar("Loss/Discriminator", d_loss, epoch)
writer.add_images("Generated Images", fake_images, epoch)
Launch TensorBoard with:

bash
Copy
Edit
tensorboard --logdir=runs
🖼️ Results
Images start as noise but become increasingly realistic.

FID scores improve with training.

TensorBoard shows loss convergence and image generation progress.

👨‍💻 Author
Vaibhav Uniyal
B.Tech Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology, Pune
GitHub: [your_username]
Email: [your_email@example.com]
