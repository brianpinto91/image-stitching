import cv2
from imgstich import utils
import argparse
from imgstich import exceptions
import os
import time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--stich_direction',
        type=int,
        required=True,
        metavar="SD",
        help="for horizontal stiching use 1 and for vertical stiching use 0"
    )

    parser.add_argument(
        '--image_folder',
        type=str,
        required=True,
        metavar="IF",
        help="folder path of the images to be stiched"
    )

    parser.add_argument(
        "--images",
        nargs="*",
        type=str,
        required=True,
        help="image filenames that are to be stiched seperated by space \
             give filenames from left to right for horizontal stiching \
             and top to bottom for vertical stiching"
    )

    parser.add_argument(
        "--save_stiched_image",
        action="store_true",
        default=False,
        help="This option will enable the saving of the stiched image"
    )

    parser.add_argument(
        "--display_result",
        action="store_true",
        default=False,
        help="This option will enable the display of the stiched image"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    num_images = len(args.images)
    
    if len(args.images) < 2:
        raise(exceptions.InsufficientImagesError(num_images))
    
    valid_files, file_error_msg = utils.check_imgfile_validity(args.image_folder, args.images)
    if not valid_files:
        raise(exceptions.InvalidImageFilesError(file_error_msg))
    
    pivot_img_path = os.path.join(args.image_folder, args.images[0])
    pivot_img = cv2.imread(pivot_img_path)

    for i in range(1, num_images, 1):
        join_img_path = os.path.join(args.image_folder, args.images[i])
        join_img = cv2.imread(join_img_path)
        pivot_img = utils.stich_images(pivot_img, join_img, stich_direc=args.stich_direction)
    
    if args.display_result:
        print("Press 0 or any key to close the image display", flush=True)
        cv2.imshow("Stiched Image", pivot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if args.save_stiched_image:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = "stiched_image_" + timestr + ".jpg"
        if not os.path.isdir("img/output"):
            os.makedirs("img/output/")
        _ = cv2.imwrite("img/output/" + filename, pivot_img)
