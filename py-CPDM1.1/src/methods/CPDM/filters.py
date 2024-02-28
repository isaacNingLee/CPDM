import torch
import torch.nn as nn
import torch.nn.functional as F

class BandPassFilter(nn.Module):
    def __init__(self, low_cutoff, high_cutoff, kernel_size=3):
        super(BandPassFilter, self).__init__()
        self.low_pass_filter = LowPassFilter(low_cutoff, kernel_size)
        self.high_pass_filter = HighPassFilter(high_cutoff, kernel_size)

    def forward(self, x):
        low_pass_output = self.low_pass_filter(x)
        high_pass_output = self.high_pass_filter(x)
        return low_pass_output - high_pass_output

# Low-pass filter
class LowPassFilter(nn.Module):
    def __init__(self, cutoff, kernel_size=3):
        super(LowPassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Same padding to keep input and output size same
        # Create a Gaussian kernel for low-pass filtering
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, cutoff)
        self.gaussian_kernel  = torch.stack([self.gaussian_kernel for _ in range(3)]).reshape(1,3,kernel_size,kernel_size).to('cuda')

    def forward(self, x):
        # Apply 2D convolution with the Gaussian kernel
        if len(x.shape) == 4:
            return F.conv2d(x, self.gaussian_kernel, padding=self.padding)
        else:
            x = x.unsqueeze(0)
            x = F.conv2d(x, self.gaussian_kernel, padding=self.padding)
            return x


    def create_gaussian_kernel(self, kernel_size, cutoff):
        """Create a 2D Gaussian kernel."""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = torch.exp(torch.tensor(-distance / (2 * (cutoff ** 2))))
        return kernel / kernel.sum()

# High-pass filter
class HighPassFilter(nn.Module):
    def __init__(self, cutoff, kernel_size=3):
        super(HighPassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Same padding to keep input and output size same
        # Create a Gaussian kernel for low-pass filtering
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, cutoff)
        self.gaussian_kernel  = torch.stack([self.gaussian_kernel for _ in range(3)]).reshape(1,3,kernel_size,kernel_size).to('cuda')

    def forward(self, x):

        if len(x.shape) == 4:
            # Convert the 3-channel image to grayscale by averaging the channels
            x_gray = x.mean(dim=1, keepdim=True)
            return x_gray - F.conv2d(x, self.gaussian_kernel, padding=self.padding)
        else:
            # Convert the 3-channel image to grayscale by averaging the channels
            x = x.unsqueeze(0)
            x_gray = x.mean(dim=1, keepdim=True)
            out = F.conv2d(x, self.gaussian_kernel, padding=self.padding)
            return (x_gray - out.squeeze(0))

    def create_gaussian_kernel(self, kernel_size, cutoff):
        """Create a 2D Gaussian kernel."""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = torch.exp(torch.tensor(-distance / (2 * (cutoff ** 2))))
        return kernel / kernel.sum()

# # Example usage:
# input_image = torch.randn(1, 3, 64, 64)  # Example input image of shape 3x64x64
# band_pass_filter = BandPassFilter(low_cutoff=5, high_cutoff=15, kernel_size=3)
# output_image = band_pass_filter(input_image)

# # The output_image will have the same dimensions as the input, containing the band-pass filtered components
# print(output_image.shape)  # Output shape: torch.Size([1, 3, 64, 64])