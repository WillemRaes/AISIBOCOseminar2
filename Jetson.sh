# Steps on Jetson Nano commandline

# The credentials for the nano are willemraes and pass is some weird bird

# Disable GUI xserver at boot
sudo systemctl set-default multi-user.target
# difference of 2 G RAM (free -m)

sudo apt-get update
sudo apt-get upgrade
# Tensorflow on Jetson Nano --> GPU support 

# 1. get the required dependencies

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# 2. Requires python3 and pip for easy install

# further requirements are
# numpy keras protobuf pybind11 gast mock but these should be installed with 

python3 -m pip install tensorflow

# As of TF v2 the main package with GPU support is just tensorflow
# Test with in python interpreter:

python3 

import tensorflow as tf
tf.config.list_physical_devices('GPU') # for GPU support verification


# Download pip wheel from location mentioned above
wget https://nvidia.box.com/shared/static/ukszbm1iklzymrt54mgxbzjfzunq7i9t.whl -O onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl

# Install pip wheel
pip3 install onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl

# Some other python packages may be required also check requirements.txt