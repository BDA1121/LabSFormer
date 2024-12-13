import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import exp
from torchvision.models import vgg16
from torchvision.transforms import Normalize
import torchvision.transforms as tr
from torchvision.models import vit_h_14
from PIL import Image


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# Function to create a 1D Gaussian window
def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian kernel for smoothing.
    Args:
        window_size (int): The size of the Gaussian window.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel.
    """
    gauss = torch.tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

# Function to create a 2D Gaussian window
def create_window(window_size, channel):
    """
    Generates a 2D Gaussian kernel by computing the outer product of 1D kernels.

    Args:
        window_size (int): Size of the Gaussian window.
        channel (int): Number of channels (applied across all channels).

    Returns:
        torch.Tensor: 2D Gaussian kernel expanded to the given channel size.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Structural Similarity Index (SSIM) computation
def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the SSIM (Structural Similarity Index) between two images.

    Args:
        img1, img2 (torch.Tensor): Input images (BxCxHxW).
        window_size (int): Size of the Gaussian kernel.
        size_average (bool): If True, return the mean SSIM value, otherwise the map.

    Returns:
        torch.Tensor: SSIM value or map.
    """
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    # Compute mean intensities using convolution

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # Compute variance and covariance
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # Constants to stabilize division
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map

# Custom loss class for SSIM
class SSIMLoss(nn.Module):
    """
    Loss function based on SSIM
    """
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, window_size=self.window_size)

# Combined Loss Function
class LABLoss(nn.Module):
    def __init__(self, ssim_weight=1, mse_weight=1):
        """
        Computes a combined loss with:
        - SSIM for L (luminance) channel.
        - MSE for A and B (chrominance) channels.
        """
        super(LABLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.mse_loss = nn.MSELoss()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        # Separate L, A, and B channels
        pred_L, pred_A, pred_B = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]
        target_L, target_A, target_B = target[:, 0:1, :, :], target[:, 1:2, :, :], target[:, 2:3, :, :]

        # SSIM loss for L channel
        loss_L = self.ssim_loss(pred_L, target_L)

        # MSE loss for A and B channels
        loss_A = self.mse_loss(pred_A, target_A)
        loss_B = self.mse_loss(pred_B, target_B)

        # Weighted sum of losses
        total_loss = self.ssim_weight * loss_L + self.mse_weight * (loss_A + loss_B)
        # print(f"SSIM Loss (L): {loss_L.item()}, MSE Loss (A): {loss_A.item()}, MSE Loss (B): {loss_B.item()}")
        return total_loss
    
# VGG-based feature extractor
class VGG16FeatureExtractor(nn.Module):
        def __init__(self):
            """
            Extracts feature maps from multiple layers of the VGG16 network.
            """
            super().__init__()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.enc_1 = nn.Sequential(*vgg16.features[:3])
            self.enc_2 = nn.Sequential(*vgg16.features[3:8])
            self.enc_3 = nn.Sequential(*vgg16.features[8:13])
            self.enc_4 = nn.Sequential(*vgg16.features[13:20])
            self.enc_5 = nn.Sequential(*vgg16.features[20:27])

            # Freezing the encoder layers
            for i in range(5):
                for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                    param.requires_grad = False

        def forward(self, image):
            # Collect features from all levels
            results = [image]
            for i in range(5):
                func = getattr(self, 'enc_{:d}'.format(i + 1))
                results.append(func(results[-1]))
            return results

# Perceptual loss based on VGG feature
class PerceptualLoss(nn.Module):
    """
    Measures perceptual differences using feature maps from a VGG16 network.
    """
    def __init__(self):
        super().__init__()
        self.extrator = VGG16FeatureExtractor()
        # Scaling factors for each layer
        self.coef = [1.0, 2.6, 4.8, 3.7, 5.6, 0.15]
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        x_out = self.extrator(x)
        y_out = self.extrator(y)
        loss = 0
        for i in range(len(x_out)):
            loss += self.l1(x_out[i], y_out[i])/self.coef[i]
        return loss


# Define the cosine similarity loss for images

class CosineSimilarityLoss(nn.Module):
    """
    Computes cosine similarity between features extracted by a VGG16 network.
    """
    def __init__(self, feature_extractor=None):
        super(CosineSimilarityLoss, self).__init__()
        # setting the feature extractir to pretrained vgg
        self.feature_extractor = vgg16(pretrained=True).features[:16] 
        self.feature_extractor.eval() 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False 
        # predefined values for vgg16
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, enhanced_image, ground_truth):
        enhanced_image = self.normalize(enhanced_image)
        ground_truth = self.normalize(ground_truth)

        # using the vgg based feature extraction
        enhanced_features = self.feature_extractor(enhanced_image)
        ground_truth_features = self.feature_extractor(ground_truth)
        # cosine_sim = F.cosine_similarity(enhanced_features, ground_truth_features, dim=1)

        # cosine sim formula
        cosine_sim=torch.mul(enhanced_features,ground_truth_features).sum(1)/(torch.mul(torch.pow((torch.pow(enhanced_features,2)).sum(1),0.5),torch.pow((torch.pow(ground_truth_features,2)).sum(1),0.5))+1e-8)
        # clamping the loss to certain range
        cosine_sim = torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7)
        loss = 1 - cosine_sim.mean() 
        return loss

# Combined loss function with Charbonnier and Cosine Similarity losses
class CombinedLoss(nn.Module):
    """
    Combines Charbonnier Loss (content loss) and Cosine Similarity Loss (feature similarity).
    """
    def __init__(self, 
                 charbonnier_weight=1.0, 
                 cosine_weight=0.5, 
                 device='cuda'):
        super(CombinedLoss, self).__init__()
        
        # Charbonnier Loss component
        self.charbonnier_loss = CharbonnierLoss()
        
        # Cosine Similarity Loss component
        self.cosine_similarity_loss = CosineSimilarityLoss(device)
        
        # Weights for each loss component
        self.charbonnier_weight = charbonnier_weight
        self.cosine_weight = cosine_weight

    def forward(self, enhanced_image, target_image):
        """
        Computes a combined loss of Charbonnier and Cosine Similarity.
        
        Parameters:
        -----------
        enhanced_image : torch.Tensor
            The enhanced (output) image.
        target_image : torch.Tensor
            The reference or target image.
        
        Returns:
        --------
        total_loss : torch.Tensor
            Weighted sum of Charbonnier and Cosine Similarity losses.
        """
        # Charbonnier Loss (pixel-wise content loss)
        charbonnier_loss = self.charbonnier_loss(enhanced_image, target_image)
        
        # Cosine Similarity Loss (feature-level similarity)
        cosine_loss = self.cosine_similarity_loss(enhanced_image, target_image)
        
        # Combined weighted loss
        total_loss = (self.charbonnier_weight * charbonnier_loss + 
                      self.cosine_weight * cosine_loss)
        
        # Optional: Log individual loss components for monitoring
        print(f"Charbonnier Loss: {charbonnier_loss.item()}, "
              f"Cosine Similarity Loss: {cosine_loss.item()}")
        
        return total_loss