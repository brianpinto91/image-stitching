from imgstich import utils
from imgstich import exceptions
import os
import cv2
import time

def stich_images(image_folder, image_filenames, stich_direction):
    """Function to stich a sequence of input images.
        Images can be stiched horizontally or vertically.
        For horizontal stiching the images have to be passed from left to right order in the scene.
        For vertical stiching the images have to be passed from top to bottom order in the scene.
    
    Args:
        image_folder (str): path of the directory containing the images
        image_filenames (list): a list of image file names in the order of stiching
        stich_direction (int): 1 for horizontal stiching, 0 for vertical stiching
    
    Returns:
        stiched_image (numpy array): of shape (H, W, 3) representing the stiched image
    """
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
        pivot_img = utils.stich_image_pair(pivot_img, join_img, stich_direc=stich_direction)
    
    return pivot_img

def stich_images_and_save(image_folder, image_filenames, stich_direction, output_folder=None):
    """Function to stich and save the resultant image.
        Images can be stiched horizontally or vertically.
        For horizontal stiching the images have to be passed from left to right order in the scene.
        For vertical stiching the images have to be passed from top to bottom order in the scene.
    
    Args:
        image_folder (str): path of the directory containing the images
        image_filenames (list): a list of image file names in the order of stiching
        stich_direction (int): 1 for horizontal stiching, 0 for vertical stiching
        output_folder (str): the directory to save the stiched image (default is None, which creates a directory named "output" to save)
    
    Returns:
        None
    """
    timestr = time.strftime("%Y%m%d_%H%M%S")
    filename = "stiched_image_" + timestr + ".jpg"
    stiched_img = stich_images(image_folder, image_filenames, stich_direction)
    if output_folder is None:
        if not os.path.isdir("output"):
            os.makedirs("output/")
            output_folder = "output"
    full_save_path = os.path.join(output_folder, filename)
    _ = cv2.imwrite(full_save_path, stiched_img)
    print("The stiched image is saved at: " + full_save_path)
