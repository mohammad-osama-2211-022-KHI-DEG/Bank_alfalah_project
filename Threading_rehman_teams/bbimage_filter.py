import os
import shutil
import tkinter as tk
import logging
from PIL import Image, ImageTk

logging.basicConfig(level=logging.DEBUG)

'''
sudo apt-get install python3-tk to install Tkinter

'''

class ImageViewer:
    def __init__(self, master, source_folder, bb_folder):
        self.master = master
        self.current_index = 0
        self.source_folder = source_folder
        self.bb_folder = bb_folder
        self.image_files = self.get_image_files(self.bb_folder)
        
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.prev_button = tk.Button(master, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(master, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.copy_button = tk.Button(master, text="Requires annotation", command=self.copy_image)
        self.copy_button.pack()

        self.show_image()

    def get_image_files(self, folder):
        logging.debug(f"Getting image files from {folder}...")
        image_files = [file for file in os.listdir(folder) if file.endswith('.jpg') or file.endswith('.png')]
        logging.debug(f"Found {len(image_files)} image files: {image_files}")
        return image_files

    def show_image(self):
        logging.debug("Displaying image...")
        image_path = os.path.join(self.bb_folder, self.image_files[self.current_index])
        image = Image.open(image_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def show_prev_image(self):
        logging.debug("Showing previous image...")
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        logging.debug("Showing next image...")
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def copy_image(self):
        logging.debug("Copying image...")
        image_filename = os.path.splitext(self.image_files[self.current_index])[0]  # Get filename without extension
        source_path = os.path.join(self.source_folder, image_filename + ".jpg")  # Assume source images have .jpg extension
        dest_folder = 'requires_annotation'
        dest_path = os.path.join(dest_folder, image_filename + ".jpg")  # Assume destination folder is 'requires annotation'

        # Create the destination folder if it doesn't exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        shutil.copy(source_path, dest_path)
        logging.info(f"Image {image_filename} copied to 'requires annotation' folder.")

def main():
    logging.debug("Starting application...")
    source_folder = input("Enter the source directory: ")
    bb_folder = input("Enter the bounding box directory: ")
    
    root = tk.Tk()
    root.title("Image Viewer")
    root.geometry("400x400")
    app = ImageViewer(root, source_folder, bb_folder)
    root.mainloop()

if __name__ == "__main__":
    main()
