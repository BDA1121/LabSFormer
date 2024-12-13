## Harshil Bhojwani and Dhanush Adithya
## CS 7180 Advanced Perception
## 12/13/2024

from PIL import Image
import os
import numpy as np
import cv2

def rgb2lab(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    
    # Apply sRGB gamma correction
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

# Example usage
# rgb_color = [255, 0, 0]  # Red
# lab_color = rgb2lab(rgb_color)
# print(lab_color)

def rgb_to_lab_image(directory_path, save_path):

    directory=os.listdir(directory_path)


    # Open the image

    for files in directory:
        
        image_path=os.path.join(directory_path, files)
        img = Image.open(image_path)
        
        # Convert image to RGB (just in case it's in a different format like RGBA)
        img_rgb = img.convert('RGB')
        
        # Convert the image to a numpy array for pixel access
        img_array = np.array(img_rgb)

        # Get the shape of the image (height, width, 3)
        height, width, _ = img_array.shape
        
        # Create an empty array to hold the LAB values
        lab_image = np.zeros((height, width, 3))
        
        # Loop through each pixel in the image
        for i in range(height):
            for j in range(width):
                
                rgb_pixel = img_array[i, j]
                lab_pixel = rgb2lab(rgb_pixel)
                lab_image[i, j] = lab_pixel

        if image_path.endswith(".png"):
            files = os.path.splitext(files)[0] + ".npy"

        save_image = os.path.join(save_path, files)

        #cv2.imwrite(save_image, lab_image)
        np.save(save_image, lab_image)
        print("Image done")


        
    return 0


def main():
     
    #rgb_to_lab_image("ShadowFormer_myimplementation/fixed_ISTD_Dataset/train/train_A","ShadowFormer_myimplementation/lab_istd_dataset/train/train_A")
    #rgb_to_lab_image("ShadowFormer_myimplementation/fixed_ISTD_Dataset/train/train_c","ShadowFormer_myimplementation/lab_istd_dataset/train/train_C")
    #rgb_to_lab_image("ShadowFormer_myimplementation/fixed_ISTD_Dataset/test/test_A","ShadowFormer_myimplementation/lab_istd_dataset/test/test_A")
    rgb_to_lab_image("ShadowFormer_myimplementation/fixed_ISTD_Dataset/test/test_c","ShadowFormer_myimplementation/lab_istd_dataset/test/test_C")




if __name__ == "__main__":
        main()
