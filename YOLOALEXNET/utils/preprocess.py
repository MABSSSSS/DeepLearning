import os
from PIL import Image

def resize_images(source_folder, dest_folder, size=(640, 640)):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in os.listdir(source_folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            img = Image.open(os.path.join(source_folder, file))
            img = img.resize(size)
            img.save(os.path.join(dest_folder, file))
