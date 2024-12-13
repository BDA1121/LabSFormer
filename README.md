
## Dhanush Adithya, Harshil Bhojwani.
## CS 7180 Advanced Perception
## 12/13/2024

OS - Linux

run pip install -r requirements.txt to set up the environment and required packages

the dataset used is ISTD [link](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view)

for testing run python test.py and optionally --save_img and --cal_metrics for saving or calculating values

for training run python train.py --warmup --win_size 8 --train_ps 256

Files edited losses.py, image_utils.py, model.py, train.py 
Files Added lab_conversion.py and istd_fixing.py

This is work is inspired from ShadowFormer[link](https://arxiv.org/pdf/2302.01650) and LaB-Net[link](https://arxiv.org/abs/2208.13039)

![image](https://github.com/user-attachments/assets/3ed08e03-d29d-4a9c-a86f-06e2a02d62c5)

![image](https://github.com/user-attachments/assets/db5392d9-2b15-43d3-b76f-938228098fc0)

