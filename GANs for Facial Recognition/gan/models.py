import torch
from gan.spectral_normalization import SpectralNorm
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.first_conv = torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding = 1)
        self.second_conv = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.third_conv = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.fourth_conv = torch.nn.Conv2d(512, 1024, kernel_size=4, stride= 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fifth_conv = torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1,  padding = 0)
        self.leaky_ReLU = nn.LeakyReLU(0.2)


    def forward(self, x):
        x = self.leaky_ReLU(self.first_conv(x))
        x = self.leaky_ReLU(self.bn1(self.second_conv(x)))
        x = self.leaky_ReLU(self.bn2(self.third_conv(x)))
        x = self.leaky_ReLU(self.bn3(self.fourth_conv(x)))
        x = self.fifth_conv(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super().__init__()    
        self.noise_dim = noise_dim
        
        self.tconv1 = nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.tconv5 = nn.ConvTranspose2d(128, 3, 4, 2, padding = 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = self.tconv5(x)
        x = self.tanh(x)

        return x