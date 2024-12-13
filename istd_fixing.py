import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
import os
from PIL import Image
import pandas as pd
import shutil


class DataConverting:

    # Initialize the DataConverting class with paths to shadow-free, shadow, and shadow-mask images
    # Also initializes paths for the directory to save processed images

    def __init__(self, shadow_free_path, shadow_path, shadow_mask_path, fixed_image_path):

        self.shadow_free_path=shadow_free_path
        self.shadow_path=shadow_path
        self.shadow_mask_path=shadow_mask_path

        # List all files in the provided directories

        self.shadow_free_dir=os.listdir(shadow_free_path)
        self.shadow_dir=os.listdir(shadow_path)
        self.shadow_mask_dir=os.listdir(shadow_mask_path)

        # Store the number of files in each directory

        self.shadow_free_path_size=len(self.shadow_free_dir)
        self.shadow_path_size=len(self.shadow_dir)
        self.shadow_mask_size=len(self.shadow_mask_dir)    

        # Path to save corrected or adjusted images
        self.fixed_image_path=fixed_image_path

    # Static method to copy contents of one directory to another
    @staticmethod
    def copy_directory(source_directory, destination_directory):

        shutil.copytree(source_directory, destination_directory, dirs_exist_ok=True)
    
    
    # Perform color adjustment on images using linear regression
    # Matches the color tone between shadow-free and shadowed images
    def color_adjustment(self,folder_name='train_C'):
         # Ensure the number of files in all directories match
        if (self.shadow_free_path_size==self.shadow_mask_size==self.shadow_path_size):

            #corrected_images=[]
            linear_data=[]
            
            saved_directory = os.path.join(self.fixed_image_path, folder_name)
            os.makedirs(saved_directory, exist_ok=True)

            print("Save directory completed")

            # Iterate over all files
            for  files in range(self.shadow_free_path_size):

                # Load shadow mask
                shadow_mask=os.path.join(self.shadow_mask_path,self.shadow_mask_dir[files])
                if not os.path.exists(shadow_mask):
                    print("mask not exists")

                shadow_mask=cv2.imread(shadow_mask)
                
                source=os.path.join(self.shadow_free_path, self.shadow_free_dir[files])

                # Load shadow-free image
                if not os.path.exists(source):
                    print("Source not exists")

                source = cv2.imread(source) 

                # Load shadowed image       
                target=os.path.join(self.shadow_path, self.shadow_dir[files])        
                if not os.path.exists(target):
                    print("target not exists")
                target = cv2.imread(target)

                # Apply the mask and normalize pixel values
                source = source[shadow_mask == 0].astype(np.float64) / 255
                target = target[shadow_mask == 0].astype(np.float64) / 255
                print("Division done")
                
                # Flatten pixel arrays for regression
                source = source.reshape(-1,3)
                target=target.reshape(-1,3)

                # Load shadow-free image again for adjustment
                shadow_free=os.path.join(self.shadow_free_path, self.shadow_free_dir[files])

                if not os.path.exists(shadow_free):
                    print("shadow_free not exists")

                shadow_free = cv2.imread(shadow_free)

                linear=[]

                # Perform linear regression for each color channel (R, G, B)
                for i in range(3):
                    reg = LinearRegression().fit(source[:, i].reshape(-1, 1), target[:, i])
                    linear.append((reg.intercept_, reg.coef_[0]))

                # Apply linear regression parameters to shadow-free image to match color tone
                
                corrected_im = shadow_free.astype(np.float64) / 255
                corrected_im[:, :, 0] = corrected_im[:, :, 0] * linear[0][1] + linear[0][0]  # Red channel
                corrected_im[:, :, 1] = corrected_im[:, :, 1] * linear[1][1] + linear[1][0]  # Green channel
                corrected_im[:, :, 2] = corrected_im[:, :, 2] * linear[2][1] + linear[2][0]  # Blue channel

                # Clip values to ensure they remain in the valid range, and convert back to uint8 format
                corrected_im = np.clip(corrected_im * 255, 0, 255).astype(np.uint8)

                image_path = os.path.join(self.fixed_image_path, f"{self.shadow_free_dir[files]}")

                print(image_path)

                # Save the corrected image
                cv2.imwrite(image_path, corrected_im)


                #corrected_images.append(corrected_im) 
                linear_data.append(linear)
            
            print("Color adjustment completed.")
            return linear_data
    
    # Calculate the percentage of saturated pixels in an image
    def calculate_percentage(self, image_path):

        image=Image.open(image_path)
        # Convert image to HSV color space
        hsv_image=image.convert("HSV")
         # Convert to NumPy array
        hsv_array=np.array(hsv_image)
        # Extract the saturation channel
        saturation_channel=hsv_array[:,:,1]

        # Total number of pixels
        total_pixel=saturation_channel.size

        # Count saturated pixels
        saturated_pixels=np.count_nonzero(saturation_channel>=255)

        # Calculate saturation percentage
        saturation_percentage=(saturated_pixels/total_pixel)*100


        return saturation_percentage
    

    # Check images for saturation and remove those exceeding the threshold
    def check_saturation(self,threshold=2):

        fixed_files=os.listdir(self.fixed_image_path)
       
        results=[]

        count=0

        # Iterate over all files in the fixed directory
        for files in fixed_files:

            file_path=os.path.join(self.fixed_image_path, files)
            
            if files.lower().endswith(('.png','.jpg','.jpeg','.bnp','.tiff')):

                saturation_percent= self.calculate_percentage(file_path)

                above_threshold=saturation_percent>threshold


                
                fixed_files.remove(files)

                self.shadow_dir.remove(files)

                self.shadow_mask_dir.remove(files)

                

                count=count+1

                print(count)

        return 0







