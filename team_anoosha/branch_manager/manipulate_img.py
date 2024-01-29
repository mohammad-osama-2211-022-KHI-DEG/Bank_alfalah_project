import os
from PIL import Image, ImageFilter
import numpy as np

def process_directory(image_folder, save_folder, process_function):
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith((".jpg", ".png", ".jpeg")):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, image_folder)  # Get the relative path
                save_subfolder = os.path.join(save_folder, os.path.dirname(relative_path))
                os.makedirs(save_subfolder, exist_ok=True)  # Create subdirectory if it doesn't exist
                process_function(file_path, save_subfolder)

def save_original(file_path, save_folder):
    img = Image.open(file_path)
    original_save_name = os.path.basename(file_path)
    img.save(os.path.join(save_folder, original_save_name))

def convert_to_gray(file_path, save_folder):
    img = Image.open(file_path)
    gray_img = img.convert('L')
    save_name = os.path.splitext(os.path.basename(file_path))[0] + "_gray" + os.path.splitext(file_path)[1]
    gray_img.save(os.path.join(save_folder, save_name))

def flip_image(file_path, save_folder):
    img = Image.open(file_path)
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    save_name = os.path.splitext(os.path.basename(file_path))[0] + "_flip" + os.path.splitext(file_path)[1]
    flipped_img.save(os.path.join(save_folder, save_name))
    convert_to_gray(os.path.join(save_folder, save_name), save_folder)

def add_noise_and_save(file_path, save_folder, mean=0, std=25):
    img = Image.open(file_path)
    noisy_img = np.array(img)
    noise = np.random.normal(mean, std, noisy_img.shape)
    noisy_img = noisy_img + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = Image.fromarray(np.uint8(noisy_img))
    noisy_save_name = os.path.splitext(os.path.basename(file_path))[0] + "_noisy" + os.path.splitext(file_path)[1]
    noisy_img.save(os.path.join(save_folder, noisy_save_name))

def blur_and_save(file_path, save_folder, radius=2):
    img = Image.open(file_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    blurred_save_name = os.path.splitext(os.path.basename(file_path))[0] + "_blurred" + os.path.splitext(file_path)[1]
    blurred_img.save(os.path.join(save_folder, blurred_save_name))

# Usage
image_folder = 'total_img'
save_folder = 'augmented_img'  # Make sure this directory exists or create it if not

# Process each subdirectory
process_directory(image_folder, save_folder, save_original)
process_directory(image_folder, save_folder, flip_image)
process_directory(image_folder, save_folder, convert_to_gray)
process_directory(image_folder, save_folder, add_noise_and_save)
process_directory(image_folder, save_folder, blur_and_save)
