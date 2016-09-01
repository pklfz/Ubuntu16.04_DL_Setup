# Ubuntu16.04_DL_Setup
Set Theano+Tensorflow+mxnet on Ubuntu16.04+CUDA8.0+cuDNNv5.1

# 从0开始搭建Ubuntu16.04深度学习工作站
### 硬件配置

> CPU: i5-3570
>
> GPU: GTX 1070
>
> RAM: 16G
>
> OS: Ubuntu16.04+Win10 Dual Boot

### 目标工作环境

> Theano

> Tensorflow
> -python3.5+CUDA8+cuDNNv5.1

> Keras

> mxnet
> -python3.5+CUDA8+cuDNNv5.1+opencv2

> optional: xgboost

### 其他软件

### 0. 安装操作系统
默认安装Ubuntu16.04,引导分区装至EFI文件系统。

### 1. 配置sudo免密码
```
sudo echo 'pklfz ALL= NOPASSWD : ALL' >> /etc/sudoers.d/nopasswd
```

### 2. 配置Grub编辑器
```
sudo add-apt-repository ppa:danielrichter2007/grub-customizer
sudo apt-get update
sudo apt-get install grub-customizer
```

### 3. 做一些关于系统的微小工作
> http://blog.csdn.net/skykingf/article/details/45267517

```
sudo apt-get remove libreoffice-common

sudo apt-get remove unity-webapps-common

sudo apt-get install vim

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt-get install libappindicator1 libindicator7
sudo dpkg -i google-chrome-stable_current_amd64.deb   
sudo apt-get -f install

sudo apt-get install vpnc git git-gui

sudo apt-get install axel

sudo apt-get install openssh-server

sudo apt-get install cmake qtcreator

sudo apt-get install exfat-fuse

sudo apt-get install unrar
```

### 4. 安装wps
1. 在`http://wps-community.org/download.html?lang=zh`下载安装包和字体包
2. 链接: http://pan.baidu.com/s/1mhCCIpE 密码: afie 下载微软字体包
3. 安装

### 5. OPTIONAL： 安装雷蛇鼠标驱动
```
下载安装包
https://bues.ch/cms/hacking/razercfg.html
sudo apt-get install libusb-1.0-0-dev
sudo apt-get install python3-pyside
cmake .
make
sudo make install
```
编辑`razer.conf`文件以适应对应鼠标
```
sudo cp razer.conf /etc/
sudo systemctl enable razerd
sudo systemctl start razerd
```

### 6. 下载科学上网客户端
```
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```

自行配置ss

gwlist规则列表
> https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt

### 7. 下载并配置realVNC方便远程访问
下载安装并启动
> https://www.realvnc.com/download/vnc/

配置开机启动
```
vim ~/.profile
# vnc
vncserver-x11 &
```

### 8. 安装atom
前往
> atom.io

常用apm列表
```
#apm install Hydrogen
apm install activate-power-mode
apm install atom-beautify
apm install atom-miku
apm install atom-monokai
apm install autocomplete-python
apm install file-icons
apm install git-time-machine
apm install highlight-selected
apm install language-ipynb
apm install markdown-writer
apm install minimap
apm install minimap-find-and-replace
apm install python-tools
apm install regex-railroad-diagram
apm install script
apm install tablr
```

## 参考以下安装深度学习工作站
> http://www.52nlp.cn/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%BB%E6%9C%BA%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE-ubuntu-16-04-nvidia-gtx-1080-cuda-8


### 9. 安装nvdia-367显卡驱动
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev
sudo nvidia-xconfig --cool-bits=28
```
第一次重启
```
reboot
```
```
sudo apt-get update
sudo apt-get upgrade
```

### 10. 安装python3.5(Anaconda)
```
wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
bash Anaconda3-4.1.1-Linux-x86_64.sh
```
选择安装环境变量

### 11. 安装CUDA8.0rc
地址：选择run local
> https://developer.nvidia.com/cuda-release-candidate-download

安装（不要安装安装包的旧驱动）
```
sudo sh cuda_8.0.27_linux.run
```

添加环境变量
```
vim ~/.bashrc
# CUDA 8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```

测试
```
nvidia-smi

cd NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery
cd ../../5_Simulations/nbody/
make
./nbody -benchmark -numbodies=256000 -device=0
```

## 参考以下继续配置工作站

> http://www.52nlp.cn/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%BB%E6%9C%BA%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE-ubuntu16-04-geforce-gtx1080-tensorflow

### 12. 配置cuDNN v5.1
下载
> https://developer.nvidia.com/rdp/cudnn-download

```
tar -zxvf cudnn-8.0-linux-x64-v5.1-ga.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

### 13. 源码编译tensorflow
安装依赖
```
sudo apt-get install python-pip
sudo apt-get install python-numpy swig python-dev python-wheel

wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-installer-linux-x86_64.sh
chmod +x bazel-0.3.0-installer-linux-x86_64.sh
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install default-jdk
./bazel-0.3.0-installer-linux-x86_64.sh --user

vim ~/.bashrc
source /home/pklfz/.bazel/bin/bazel-complete.bash
export PATH=$PATH:/home/pklfz/.bazel/bin
source ~/.bashrc
```

tensorflow
```
git clone https://github.com/tensorflow/tensorflow
sudo apt-get install libcurl3 libcurl3-dev zlib1g-dev
./configure
选择y和默认配置
```

<!--*注意*：-->
<!--在`third_party/gpus/crosstool/CROSSTOOL`中寻找`cxx_builtin_include_directory`，并追加-->
<!--`cxx_builtin_include_directory: "/usr/local/cuda-8.0/include"`-->

编译，较慢
```
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
```

测试
```
bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
```

打包whl
```
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/
pip install ~/tensorflow-0.10.0rc0-py3-none-any.whl
```

### 14. 安装theano
```
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
sudo apt-get install libblas-dev
sudo apt-get install libopencv-dev
pip install nose-parameterized
```

`~/.theanorc`配置文件
```
[global]
openmp=False
device = gpu
optimizer=fast_run
optimizer_including=cudnn
floatX = float32
allow_input_downcast=True
[blas]
ldflags=-lblas
[gcc]
#cxxflags=
[nvcc]
fastmath = True
#flags =
#compiler_bindir =
flags=-arch=sm_60
[lib]
#cnmem=.75
```

### 15. 安装keras
```
pip install keras
conda install graphviz
wget https://github.com/erocarrera/pydot/archive/v1.1.0.tar.gz
tar -zxvf v1.1.0.tar.gz
cd v1.1.0
2to3 -w *.py # 可能仍有py2和py3不兼容的地方，如果运行时pydot报错，需要卸载pydot并更改兼容性后重新安装
python setup.py install
```

### 16. 安装opencv2
Ubuntu16.04 apt-get到的opencv是3版本，不兼容mxnet，故手工编译opencv2

参考
> https://gist.github.com/dynamicguy/3d1fce8dae65e765f7c4

```
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y cmake
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y pkg-config
sudo apt-get install -y python-numpy python-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libjasper-dev
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2 libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
```
```
wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.11/opencv-2.4.11.zip
unzip opencv-2.4.11.zip
cd opencv-2.4.11
mkdir release
cd release
```
```
cmake -G "Unix Makefiles" -D CMAKE_CXX_COMPILER=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_FAT_JAVA_LIB=ON -D INSTALL_TO_MANGLED_PATHS=ON -D INSTALL_CREATE_DISTRIB=ON -D INSTALL_TESTS=ON -D ENABLE_FAST_MATH=ON -D WITH_IMAGEIO=ON -D BUILD_SHARED_LIBS=OFF -D WITH_GSTREAMER=ON -D WITH_CUDA=OFF ..
make all -j4 # 4 cores
sudo make install
```

### 17. 安装mxnet
> http://mxnet.readthedocs.io/en/latest/how_to/build.html

```
cp make/config.mk .
修改 config.mk 使用CUDA
```
```
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; make -j$(nproc)
```
安装python
```
cd python
python setup.py install
cd ..
```

### 18. 验证环境
1. keras

  脚本`test_keras_mnist.py`

  ```
  '''Trains a simple convnet on the MNIST dataset.

  Gets to 99.25% test accuracy after 12 epochs
  (there is still a lot of margin for parameter tuning).
  16 seconds per epoch on a GRID K520 GPU.
  '''

  from __future__ import print_function
  import numpy as np
  np.random.seed(1337)  # for reproducibility

  from keras.datasets import mnist
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Activation, Flatten
  from keras.layers import Convolution2D, MaxPooling2D
  from keras.utils import np_utils

  batch_size = 128
  nb_classes = 10
  nb_epoch = 12

  # input image dimensions
  img_rows, img_cols = 28, 28
  # number of convolutional filters to use
  nb_filters = 32
  # size of pooling area for max pooling
  nb_pool = 2
  # convolution kernel size
  nb_conv = 3

  # the data, shuffled and split between train and test sets
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255
  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(y_train, nb_classes)
  Y_test = np_utils.to_categorical(y_test, nb_classes)

  model = Sequential()

  model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                          border_mode='valid',
                          input_shape=(1, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

  model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(X_test, Y_test))
  score = model.evaluate(X_test, Y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  ```

  1. Theano

    `KERAS_BACKEND=theano python test_keras_mnist.py`

  2. Tensorflow

    `KERAS_BACKEND=tensorflow python test_keras_mnist.py`

2. mxnet
```
cd MXNET_HOME
python example/image-classification/train_mnist.py --network lenet --gpu 0
python example/image-classification/train_cifar10.py
```

### 19. OPTIONAL: xgboost
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; python setup.py install
```
