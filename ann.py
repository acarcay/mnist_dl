import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data_loaders(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

def visualize_samples(loader, n):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, n, figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')
    plt.show()

visualize_samples(train_loader, 4)