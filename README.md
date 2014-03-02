cuda-convnet
============

A fork of cuda-convnet.

(The original website and codes are out-of-maintenance, and it also legacy codes only in CUDA 4.0 )

# Prerequisite

  - NVIDIA CUDA Toolkit (5.0 at least)
  - Python development libraries/headers
  - Numpy
  - Python libmagic bindings
  - Matplotlib
  - ATLAS development libraries/headers

```bash
sudo easy_install python-dev
sudo easy_install python-numpy
sudo easy_install python-magic
sudo easy_install python-matplotlib
sudo apt-get install libatlas-base-dev
```

# How to Install
First of all, make sure all of the above packages are installed. Then
```bash
./install-sh
```
# How to Use
Just run ```./go_example.sh```. It will first download CIFAR-10 dataset from http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz and then use it to train a convolutional neural network.
