from imgstich import utils
from imgstich import exceptions
import os
import cv2

def stich_images(image_folder, image_filenames, stich_direction):
    num_images = len(image_filenames)
    
    if num_images < 2:
        raise(exceptions.InsufficientImagesError(num_images))
    
    valid_files, file_error_msg = utils.check_imgfile_validity(image_folder, image_filenames)
    if not valid_files:
        raise(exceptions.InvalidImageFilesError(file_error_msg))
    
    pivot_img_path = os.path.join(image_folder, image_filenames[0])
    pivot_img = cv2.imread(pivot_img_path)

    for i in range(1, num_images, 1):
        join_img_path = os.path.join(image_folder, image_filenames[i])
        join_img = cv2.imread(join_img_path)
        pivot_img = utils.stich_images(pivot_img, join_img, stich_direc=stich_direction)
    
    return pivot_img