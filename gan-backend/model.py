import torch
from torch import nn
import torchvision
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os


generator_directory = './models/ganGeneratorv55.pt'
n_classes = 9
mnist_shape = (1, 28, 28)
device = 'cpu'
z_dim = 64

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise (n_samples, input_dim, device='cpu'):
  return torch.randn(n_samples, input_dim, device = device)

def combine_vectors(x, y):
    combined = torch.cat((x.float(),y.float()), 1)
    return combined

def get_input_dimensions(z_dim, mnist_shape, n_classes):

    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan

def show_tensor_images(image_tensor, num_images, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.figure(figsize=(16, 16))
    if show:
        plt.show()


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels,n_classes)

def show_tensor_images(image_tensor, num_images, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.figure(figsize=(16, 16))
    if show:
        plt.show()
    plt.savefig("./images/test.png", bbox_inches = "tight", pad_inches = 0.0)


def get_model():
    generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)
    gen = Generator(input_dim=generator_input_dim).to(device)
    gen.load_state_dict(torch.load(generator_directory,map_location=torch.device('cpu')))
    gen.eval()
    return gen

def get_images(num_of_examples, label, gen, save_dir):
    num_of_examples = num_of_examples ## INPUT

    label = label ## INPUT this can be in the range of 0 to 39, 0 is alif, 1 is alif mad aa and so on i think this is in ascending order but im not sure

    fake_noise = get_noise(num_of_examples, z_dim, device=device)
    one_hot_labels = get_one_hot_labels(torch.Tensor([label]).long(), n_classes).repeat(num_of_examples, 1)
    noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
    fake = gen(noise_and_labels)

    os.mkdir(save_dir)

    for i in range(0, num_of_examples):
        image = torchvision.transforms.Resize(96)(fake[i])
        save_image(image, f'{save_dir}/image-{i}.png')

def getCategories():
    return ['alif', 'ayn', 'baa', 'daal', 'hamza', 'meem', 'seen', 'swaad', 'twa']
