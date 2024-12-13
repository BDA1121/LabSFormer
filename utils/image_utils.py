import torch
import warnings
import numpy as np
import pickle
import cv2
from skimage.color import rgb2lab, lab2rgb

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def rgb2lab(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    
    # Applying sRGB gamma correction
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # Convert to XYZ color space
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    # Apply XYZ to Lab conversion
    x = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    y = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    z = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    # Return Lab components
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return [L, a, b]

# Loads an image, converts it to Lab color space, and normalizes it
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    lab_img = rgb2lab(img / 255.0)
    # Normalize Lab channels
    normalized_lab = np.empty_like(lab_img, dtype=np.float32)
    # as L range is 0-100 we divide it by 100
    normalized_lab[..., 0] = lab_img[..., 0] / 100.0
    # as A range is -128 to 127 we add 128 and divide by 255 to bring it to 0 to 1
    normalized_lab[..., 1] = (lab_img[..., 1] + 128.0) / 255.0
    # as B range is -128 to 127 we add 128 and divide by 255 to bring it to 0 to 1

    normalized_lab[..., 2] = (lab_img[..., 2] + 128.0) / 255.0

    return normalized_lab

# a similar function as above but for validation images, Loads an image, converts it to Lab color space, and normalizes it
def load_val_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    lab_img = rgb2lab(img / 255.0)
    # Normalize Lab channels

    normalized_lab = np.empty_like(lab_img, dtype=np.float32)
    # as L range is 0-100 we divide it by 100

    normalized_lab[..., 0] = lab_img[..., 0] / 100.0
    # as A range is -128 to 127 we add 128 and divide by 255 to bring it to 0 to 1

    normalized_lab[..., 1] = (lab_img[..., 1] + 128.0) / 255.0
    # as B range is -128 to 127 we add 128 and divide by 255 to bring it to 0 to 1

    normalized_lab[..., 2] = (lab_img[..., 2] + 128.0) / 255.0
    return normalized_lab

# Loads the mask image in grayscale and normalizes it (mask only contains two values 0 and 255 where 255 is for the shadow region) to 0 and 1.
def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img

# similar function as above but for validation,- Loads the mask image in grayscale and normalizes it (mask only contains two values 0 and 255 where 255 is for the shadow region) to 0 and 1.
def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = img
    resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

# function for saving image after testing
def save_img(lab_img, filepath):
    # LAB images need to be denormalized before passing converting them into rgb to have no loss of value
    denormalized_lab = np.empty_like(lab_img, dtype=np.float32)
    # Denormalize L channel
    denormalized_lab[..., 0] = lab_img[..., 0] * 100.0

# Denormalize A channel
    denormalized_lab[..., 1] = lab_img[..., 1] * 255.0 - 128.0

# Denormalize B channel
    denormalized_lab[..., 2] = lab_img[..., 2] * 255.0 - 128.0

# # Convert LAB to RGB
    rgb_img = lab2rgb(denormalized_lab)

# save the image
    img_np = (rgb_img * 255).astype(np.uint8) 
    cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# function to convert lab to rgb for psnr calculation during training
def psnrlabrgb(img):
    img_lab = img.cpu().detach().numpy()

    # Permute tensor from (C, H, W) to (H, W, C)
    img_lab = np.transpose(img_lab, (1, 2, 0))

    # Denormalize LAB values
    img_lab[:, :, 0] *= 100.0  # Denormalize L to [0, 100]
    img_lab[:, :, 1] = img_lab[:, :, 1] * 255.0 - 128.0  # Denormalize A to [-128, 127]
    img_lab[:, :, 2] = img_lab[:, :, 2] * 255.0 - 128.0  # Denormalize B to [-128, 127]
    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0, 100)
    img_lab[:, :, 1] = np.clip(img_lab[:, :, 1], -128, 127)
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2], -128, 127)

    # Convert LAB to RGB ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) 
        img_rgb = lab2rgb(img_lab) 
    # Convert back to tensor and permute to (C, H, W)
    img_rgb_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()

    return img_rgb_tensor


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        im1 = psnrlabrgb(im1)
        im2 = psnrlabrgb(im2)
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

# def yCbCr2rgb(input_im):
#     im_flat = input_im.contiguous().view(-1, 3).float()
#     mat = torch.tensor([[1.164, 1.164, 1.164],
#                        [0, -0.392, 2.017],
#                        [1.596, -0.813, 0]])
#     bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
#     temp = (im_flat + bias).mm(mat)
#     out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])