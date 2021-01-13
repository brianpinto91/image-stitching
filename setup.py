import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imgstich",
    version="0.0.1",
    author="Brian Pinto",
    author_email="brian.pinto@tuhh.de",
    description="A python package that can be used to stich multiple images either horizontally or vertically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brianpinto91/image-stiching",
    packages=setuptools.find_packages(),
    install_requires=[
          'opencv-python>=4.5.1.48', 'numpy>=1.19.4',
      ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
)