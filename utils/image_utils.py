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

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = img.astype(np.float32)
    # img = img/255.
    # img = np.log1p(img)
    lab_img = rgb2lab(img / 255.0)
    normalized_lab = np.empty_like(lab_img, dtype=np.float32)
    normalized_lab[..., 0] = lab_img[..., 0] / 100.0
    normalized_lab[..., 1] = (lab_img[..., 1] + 128.0) / 255.0
    normalized_lab[..., 2] = (lab_img[..., 2] + 128.0) / 255.0

    
#     lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
#     lab_img[:, :, 0] /= 100.0  # Normalize L to [0, 1]
#     lab_img[:, :, 1] = (lab_img[:, :, 1] + 128.0) / 255.0  # Normalize A to [0, 1]
#     lab_img[:, :, 2] = (lab_img[:, :, 2] + 128.0) / 255.0  
#     lab_img_1 = lab_img
#     lab_img_1[:, :, 0] *= 100.0  # Back to [0, 100]
#     lab_img_1[:, :, 1] = lab_img_1[:, :, 1] * 255.0 - 128.0  # Back to [-128, 127]
#     lab_img_1[:, :, 2] = lab_img_1[:, :, 2] * 255.0 - 128.0  # Back to [-128, 127]

# # Convert LAB to RGB
#     rgb_img = cv2.cvtColor(lab_img_1.astype(np.float64), cv2.COLOR_LAB2RGB)
#     rgb_img = np.clip(rgb_img, 0, 1)  # Clip values to [0, 1]

# # Optionally save the image
#     img_np = (rgb_img * 255).astype(np.uint8) 
#     cv2.imwrite("test.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)) 

# Merge channels back
    # img = cv2.merge([L_normalized, A_normalized, B_normalized])
    # cv2.imwrite("test.png", lab_img)
    return normalized_lab

def load_val_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # resized_img = img.astype(np.float32)
    # # resized_img = resized_img/255.
    # # img = np.log1p(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # L, A, B = cv2.split(img)

# Normalize each channel
    # L_normalized = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # A_normalized = cv2.normalize(A, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # B_normalized = cv2.normalize(B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    # lab_img[:, :, 0] /= 100.0  # Normalize L to [0, 1]
    # lab_img[:, :, 1] = (lab_img[:, :, 1] + 128.0) / 255.0  # Normalize A to [0, 1]
    # lab_img[:, :, 2] = (lab_img[:, :, 2] + 128.0) / 255.0  
    lab_img = rgb2lab(img / 255.0)
    normalized_lab = np.empty_like(lab_img, dtype=np.float32)
    normalized_lab[..., 0] = lab_img[..., 0] / 100.0
    normalized_lab[..., 1] = (lab_img[..., 1] + 128.0) / 255.0
    normalized_lab[..., 2] = (lab_img[..., 2] + 128.0) / 255.0
# Merge channels back
    # img = cv2.merge([L_normalized, A_normalized, B_normalized])
    # cv2.imwrite("test.png", lab_img)

    return normalized_lab

def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = img
    resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def save_img(lab_img, filepath):
    denormalized_lab = np.empty_like(lab_img, dtype=np.float32)
    denormalized_lab[..., 0] = lab_img[..., 0] * 100.0

# Denormalize A channel
    denormalized_lab[..., 1] = lab_img[..., 1] * 255.0 - 128.0

# Denormalize B channel
    denormalized_lab[..., 2] = lab_img[..., 2] * 255.0 - 128.0

# # Convert LAB to RGB
    rgb_img = lab2rgb(denormalized_lab)
    # rgb_img = np.clip(rgb_img, 0, 1)  # Clip values to [0, 1]

# Optionally save the image
    img_np = (rgb_img * 255).astype(np.uint8) 
    cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def psnrlabrgb1(img):
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

    # Convert LAB to RGB using skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) 
        img_rgb = lab2rgb(img_lab)  # Output is normalized to [0, 1] by skimage
    img_np = (img_rgb * 255).astype(np.uint8) 
    cv2.imwrite("sample.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Convert back to tensor and permute to (C, H, W)
    img_rgb_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()

    return img_rgb_tensor

def psnrlabrgb2(img):
    img_lab = img.cpu().detach().numpy()

    # Permute tensor from (C, H, W) to (H, W, C)
    img_lab = np.transpose(img_lab, (1, 2, 0))

    # Denormalize LAB values
    img_lab[:, :, 0] *= 100.0  # Denormalize L to [0, 100]
    img_lab[:, :, 1] = img_lab[:, :, 1] * 255.0 - 128.0  # Denormalize A to [-128, 127]
    img_lab[:, :, 2] = img_lab[:, :, 2] * 255.0 - 128.0  # Denormalize B to [-128, 127]

    # Convert LAB to RGB using skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) 
        img_rgb = lab2rgb(img_lab)  # Output is normalized to [0, 1] by skimage
    img_np = (img_rgb * 255).astype(np.uint8) 
    cv2.imwrite("sample1.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

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
        im1 = psnrlabrgb1(im1)
        im2 = psnrlabrgb2(im2)
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