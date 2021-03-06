import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    criterion_1 = bce_loss(logits_real, torch.ones_like(logits_real).cuda())
    criterion_2 = bce_loss(logits_fake, torch.zeros_like(logits_fake).cuda())
    
    loss = criterion_1 + criterion_2
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake).cuda())
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    loss = torch.mean(torch.square(scores_real - 1)) + torch.mean(torch.square(scores_fake))
    
    return loss/2

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = torch.mean(torch.square(scores_fake - 1))
    
    return loss/2