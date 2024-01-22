import os
from PIL import Image

def convert_rgb_to_gray(image_folder, save_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(image_folder, filename))
            gray_img = img.convert('L')
            save_name = os.path.splitext(filename)[0] + "_gray" + os.path.splitext(filename)[1]
            gray_img.save(os.path.join(save_folder, save_name))

# Usage
convert_rgb_to_gray('Images/guard/', 'Images/guard/')

def flip_and_save_images(image_folder, save_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(image_folder, filename))
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            save_name = os.path.splitext(filename)[0] + "_flip" + os.path.splitext(filename)[1]
            flipped_img.save(os.path.join(save_folder, save_name))

# Usage
flip_and_save_images('Images/guard/', 'Images/guard/')
