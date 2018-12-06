# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).This  code is from zhixuhao

---

## Overview

### Data

The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), and I've downloaded it and done the pre-processing.

You can find it in folder data/membrane.

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

See dataPrepare.ipynb and data.py for detail.


### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.


---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0

Also, this code should be compatible with Python versions 2.7-3.5.

### Run main.py

You will see the predicted results of test image in data/membrane/test

### Or follow notebook trainUnet



### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
# keras_unet
讲解来自U-net源码讲解（Keras）[https://blog.csdn.net/mieleizhi0522/article/details/82217677]
源码文件夹目录：



这里主要讲解data.py,  model.py,   main.py三个文件（也只要这三个python文件）

先看一下main.py，按照main.py文件的运行顺序去查找每个函数的意义：

from model import *
from data import *#导入这两个文件中的所有函数
 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
 
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')#数据增强时的变换方式的字典
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
#得到一个生成器，以batch=2的速率无限生成增强后的数据
 
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#回调函数，第一个是保存模型路径，第二个是检测的值，检测Loss是使它最小，第三个是只保存在验证集上性能最好的模型
 
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
#steps_per_epoch指的是每个epoch有多少个batch_size，也就是训练集总样本数除以batch_size的值
#上面一行是利用生成器进行batch_size数量的训练，样本和标签通过myGene传入
testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#30是step,steps: 在停止之前，来自 generator 的总步数 (样本批次)。 可选参数 Sequence：如果未指定，将使用len(generator) 作为步数。
#上面的返回值是：预测值的 Numpy 数组。
saveResult("data/membrane/test",results)#保存结果
data.py文件：

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
 
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
 
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
 
 
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):#此程序中不是多类情况，所以不考虑这个
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
#if else的简洁写法，一行表达式，为真时放在前面，不明白mask.shape=4的情况是什么，由于有batch_size，所以mask就有3维[batch_size,wigth,heigh],估计mask[:,:,0]是写错了，应该写成[0,:,:],这样可以得到一片图片，
        new_mask = np.zeros(mask.shape + (num_class,))
#np.zeros里面是shape元组，此目的是将数据厚度扩展到num_class层，以在层的方向实现one-hot结构
 
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1#将平面的mask的每类，都单独变成一层，
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
#上面这个函数主要是对训练集的数据和标签的像素值进行归一化
 
 
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(#https://blog.csdn.net/nima1994/article/details/80626239
        train_path,#训练数据文件夹路径
        classes = [image_folder],#类别文件夹,对哪一个类进行增强
        class_mode = None,#不返回标签
        color_mode = image_color_mode,#灰度，单通道模式
        target_size = target_size,#转换后的目标图片大小
        batch_size = batch_size,#每次产生的（进行转换的）图片张数
        save_to_dir = save_to_dir,#保存的图片路径
        save_prefix  = image_save_prefix,#生成图片的前缀，仅当提供save_to_dir时有效
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)#组合成一个生成器
    for (img,mask) in train_generator:
#由于batch是2，所以一次返回两张，即img是一个2张灰度图片的数组，[2,256,256]
        img,mask = adjustData(img,mask,flag_multi_class,num_class)#返回的img依旧是[2,256,256]
        yield (img,mask)
#每次分别产出两张图片和标签，不懂yield的请看https://blog.csdn.net/mieleizhi0522/article/details/82142856
 
#上面这个函数主要是产生一个数据增强的图片生成器，方便后面使用这个生成器不断生成图片
 
 
def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
#将测试图片扩展一个维度，与训练时的输入[2,256,256]保持一致
        yield img
 
#上面这个函数主要是对测试图片进行规范，使其尺寸和维度上和训练图片保持一致
 
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
#相当于文件搜索，搜索某路径下与字符匹配的文件https://blog.csdn.net/u010472607/article/details/76857493/
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):#enumerate是枚举，输出[(0,item0),(1,item1),(2,item2)]
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
#重新在mask_path文件夹下搜索带有mask字符的图片（标签图片）
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)#转换成array
    return image_arr,mask_arr
#该函数主要是分别在训练集文件夹下和标签文件夹下搜索图片，然后扩展一个维度后以array的形式返回，是为了在没用数据增强时的读取文件夹内自带的数据
 
 
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
#变成RGB空间，因为其他颜色只能再RGB空间才会显示
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
#为不同类别涂上不同的颜色，color_dict[i]是与类别数有关的颜色，img_out[img == i,:]是img_out在img中等于i类的位置上的点
    return img_out / 255
 
#上面函数是给出测试后的输出之后，为输出涂上不同的颜色，多类情况下才起作用，两类的话无用
 
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#多类的话就图成彩色，非多类（两类）的话就是黑白色
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
这里要说明一下，由于在预测的时候模型是直接输出的，下面模型的输出是在一个sigmoid函数之后的输出，也就是输出的数值是在0-1之间的，但是在这里直接就把这个0-1之间的数进行保存成图片了，这里有两个疑点：

1.为什么可以直接将在0-1的浮点数直接保存成图片？

是因为在skimage模块中，如果图片数据是float的话，那么值应该是0到1或者-1到1的浮点数，

2.为什么直接保存而不进行mask二值图像的产生？

这是因为输出数据值已经很两极分化了，也即是有的很接近于0，有的很接近于1了，中间的数值很少，所以就直接输出也没有关系，相当于输出的是灰度图，如果你感觉非要产生二值化图像，可以修改成下面代码：

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
#多类的话就图成彩色，非多类（两类）的话就是黑白色
        else:
            img=item[:,:,0]
            print(np.max(img),np.min(img))
            img[img>0.5]=1#此时1是浮点数，下面的0也是
            img[img<=0.5]=0
            print(np.max(img),np.min(img))
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
下面是model.py：

 

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
 
 
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
 
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
 
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))#上采样之后再进行卷积，相当于转置卷积操作！
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
 
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)#我怀疑这个sigmoid激活函数是多余的，因为在后面的loss中用到的就是二进制交叉熵，包含了sigmoid
 
    model = Model(input = inputs, output = conv10)
 
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])#模型执行之前必须要编译https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
    #利用二进制交叉熵，也就是sigmoid交叉熵，metrics一般选用准确率，它会使准确率往高处发展
    #model.summary()
 
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
 
    return model
 
 
到此结束：

看看测试的结果：



1.你会发现测试的输出是256*256，但是输入是512*512，这是因为在输入的时候被resize了，统一resize成256*256.

2.还有一个就是这个模型没有按照论文中的模型来创建，具体区别就是每次卷积的时候这里采用的是padding=same,而论文中是没有进行pad的，也就是这里的输入尺寸和输出尺寸是一样大的，而论文中是输入大于输出。具体请看

如果大家有爱好深度学习，爱好人工智能，可以加下我创建的群825524664（深度学习交流），仅供学习交流，没有广告，谢谢大家捧场！

 

全黑或者全灰的解决方法：
1.尽量用python3去跑

2.img/255的地方全部改成img/255.0试试

