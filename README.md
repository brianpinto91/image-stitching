# A python package to stich multiple images

[![GitHub repo size](https://img.shields.io/github/repo-size/brianpinto91/image-stiching?logo=GitHub)]()
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/brianpinto91/image-stiching/main)](https://www.codefactor.io/repository/github/brianpinto91/image-stiching)

This package provides an elegant function to stich images from a scene either horizontally or vertically.

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Implementation](#Model-training-on-GoogleCloud)
* [License](#license)

## Installation

Currently the package can be installed from TestPyPi python software repository. Once the project is finalized it will be available on PyPi.

To install the package, you can use the **PIP** installer for python using the command:

```sh
$ pip install -i https://test.pypi.org/simple/ imgstich
```

## Usage

There are two functions that can be primarily used:
 
 #### `stich_images(image_folder, image_filenames, stich_direction)`

 Args:

 - **`image_folder`**  `str` `required`
    
    path of the directory containing the images
 - **`image_filenames`**  `list` `required`
    
    a list of image file names in the order of stiching.
 - **`stich_direction`**  `int` `required`
    
    direction of stiching. Uses numpy convention. Use 0 for stiching along image height or vertical stiching. And use 1 for stiching along image width or horizontal stiching.

Returns:
 - **`stiched_image`**  `numpy array`

    a numpy array of representing the stiched image. The channels will be in the opencv convention that is BGR.

#### `stich_images_and_save(image_folder, image_filenames, stich_direction, output_folder)`

 Args:

 - **`image_folder`**  `str` `required`
    
    path of the directory containing the images
 - **`image_filenames`**  `list` `required`
    
    a list of image file names in the order of stiching.
 - **`stich_direction`**  `int` `required`
    
    direction of stiching. Uses numpy convention. Use 0 for stiching along image height or vertical stiching. And use 1 for stiching along image width or horizontal stiching.
 - **`output_folder`**  `str`  `optional` `default = "output"`

    the directory to which the stiched image is to be saved. By default a directory named "output" is created in the parent directory of your python script which uses this function. 

Returns:
 - **`None`**

    The image is saved in the specified directiory or the default directory with a time stamp attached to the filename (stiched_image_yyyymmdd_hhmmss.jpg)

