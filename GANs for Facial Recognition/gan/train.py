import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for the losses.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """

    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
            transforms = torch.nn.Sequential(transforms.RandomHorizontalFlip(p=0.75), transforms.RandomHorizontalFlip(p=0.5), transforms.FiveCrop(5))
            
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None

            #Discriminator Training

            D_solver.zero_grad()
            noise = sample_noise(batch_size, noise_size).view(batch_size, noise_size, 1, 1).cuda()
            d_output_real = D(real_images)
            fake_images = G(noise).cuda().detach()
            
            #Disc Loss
            d_error = discriminator_loss(d_output_real.view(-1,), D(fake_images).view(-1,))
            #Gradient Computations
            d_error.backward()
            #Optimizer Steps
            D_solver.step()
            
            #Generator Training

            G_solver.zero_grad()
            g_noise = sample_noise(batch_size, noise_size).view(batch_size, noise_size, 1, 1).cuda()
            fake_images = G(g_noise).cuda().detach()

            #Generator Loss
            g_error = generator_loss(D(fake_images).view(-1,))
            #Gradient Computations
            g_error.backward()
            #Optimizer Steps
            G_solver.step()
            
            
            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1