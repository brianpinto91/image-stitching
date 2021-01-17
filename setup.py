""" imgstich is a python package to stich multiple images

It provides:

- a function to join a list of images corressponding to a scene by passing image filenames in an order
- a function to stich images along the width as well as height, but not together
- a function to save the stiched image or return it as an numpy array

For detailed information please visit the links for Homepage or Documetation on the left hand side under Project links.  
"""


DOCLINES = (__doc__ or '').split("\n")

import setuptools

setuptools.setup(
    name="imgstich",
    version="0.0.6",
    author="Brian Pinto",
    author_email="brian.pinto@tuhh.de",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://brianpinto91.github.io/image-stiching",
    project_urls={
            "Bug Tracker": "https://github.com/brianpinto91/image-stiching/issues",
            "Documentation": "https://brianpinto91.github.io/image-stiching",
            "Source Code": "https://github.com/brianpinto91/image-stiching",
        },
    packages=setuptools.find_packages(),
    install_requires=[
          'opencv-python>=4.5.1.48', 'numpy>=1.19.4',
      ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
)