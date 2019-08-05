from __future__ import division
from keras import layers
from keras.models import Model
from keras.layers import Lambda,Conv2D,LSTM, Concatenate,GlobalMaxPooling2D,GlobalAveragePooling2D

from keras.layers.core import Dropout, Activation,Dense,Reshape,Flatten
from keras.layers import Input, merge,Add,multiply, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D ,Deconvolution2D,ZeroPadding2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
#from keras.layers import Conv2D, Concatenate, MaxPooling2D
#from keras.layers.convolutional import AtrousConvolution2D
from keras.utils.data_utils import get_file

#from keras.regularizers import l2
import keras.backend as K
#import theano.tensor as T
#from PCreshape import PCreshape
#from keras.applications.inception_v3 import InceptionV3, conv2d_bn

import scipy.io
import scipy.ndimage
#import h5py
#from eltwise_product import EltWiseProduct
#from config2 import *
#from subpixel import PS
import tensorflow as tf
import numpy as np
from keras import applications

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH_NO_TOP2 = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

def TVdist(y_true, y_pred, eps=K.epsilon()):
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        TVdist = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        # return kld*0.5
        return TVdist*0.5



class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)



def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True,dilation_rate=(1, 1), train_bn=None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), use_bias=use_bias, name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,dilation_rate=dilation_rate,
               padding='same',use_bias=use_bias,  name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),use_bias=use_bias,  name=conv_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, strides=(2, 2),dilation_rate=(1, 1), train_bn=None):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,use_bias=use_bias, 
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, dilation_rate=dilation_rate, padding='same',use_bias=use_bias, 
               name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=use_bias, name=conv_name_base + '2c')(x)
    #x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,use_bias=use_bias, 
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNorm(axis=bn_axis, name=bn_name_base + '1')(shortcut, training=train_bn)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def kerasResnet(input_tensor=None, train_bn=None):
    input_shape = (None, None,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    bn_axis= -1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    model = Model(img_input, x)

    # Load weights
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',  WEIGHTS_PATH_NO_TOP2,
                           cache_subdir='models',md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)

    return model

def M_VGG16(input_tensor=None):
    input_shape = (None, None,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

#    channel_axis = -1
    
    conv1_1 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(img_input)
    conv1_2 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    # conv_2
    conv2_1 = Convolution2D(128,kernel_size=(3, 3), activation='relu', padding='same')(conv1_pool)
    conv2_2 = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    # conv_3
    conv3_1 = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv2_pool)
    conv3_2 = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv3_3)

    # conv_4
    conv4_1 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv3_pool)
    conv4_2 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_1)
    conv4_3 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv4_3)

    # conv_5
    conv5_1 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(1, 1))(conv4_pool)
    conv5_2 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(1,1))(conv5_1)
    x = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(1, 1))(conv5_2)
#    concatenated = merge([conv3_pool, conv4_pool, conv5_3], mode='concat', concat_axis=1)
    
    model = Model(img_input, x, name='M_VGG16')
#    model1 = Model(inputs=[input_ml_net], outputs=[x])
    
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    
    model.load_weights(weights_path)

    return model


def se_block(input_tensor, compress_rate = 16):
    num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
    bottle_neck = int(num_channels//compress_rate)
 
    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch0 = Dense(bottle_neck, activation='relu')(se_branch)
    se_branch = Dense(num_channels, activation='sigmoid')(se_branch0)
 
    x = input_tensor 
    out = multiply([x, se_branch])
 
    return out,se_branch0


#model defination

def ktresnetbaselineplusavg2ms(img_rows=480, img_cols=640, fixsed =False,out1dim=512,out2dim=512, train_bn=None, load =False):

    input_ml_net = Input(shape=(img_rows, img_cols,3))
    
#    input_ml_net2 = Input(shape=(int(img_rows/8), int(img_cols/8),1))
    base_model = kerasResnet(input_tensor=input_ml_net, train_bn=None)

    if fixsed:
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output

    x1 = Convolution2D(out1dim, kernel_size=(1, 1), activation='relu',padding='same',use_bias=False)(x) 
    x,_= se_block(x1, compress_rate = 4)   #this is right
    #x = BatchNormalization(axis=-1, scale=False)(x)

    sm = Convolution2D(1, kernel_size=(1, 1), activation='relu',padding='same',use_bias=False,name='saliency')(x1) 

    # attention1z = Lambda(repeat3, arguments={'num':out1dim})(sm)
    # print attention1z.shape
    x=layers.multiply([x,sm])
    # x= se_block(x, compress_rate = 4)
    x= GlobalAveragePooling2D()(x)
    # sm2 = Convolution2D(2, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False,name='saliency0')(sm) 
    # sm2 = Flatten()(sm2)
    # x=layers.concatenate([x,sm2],axis=-1)

    #x= GlobalMaxPooling2D()(x)
#    x = Dropout(0.5)(x)
    x = Dense(out2dim, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(out2dim, activation='relu', name='fc2')(x)
#    x = Dropout(0.5)(x)
    final_output= Dense(1,name='predictions')(x)
#    , activation='softmax', name='predictions'
#   
#    
    if load:
        model = Model(inputs=[input_ml_net], outputs=[final_output])
    else: 
        model = Model(inputs=[input_ml_net], outputs=[final_output,sm])    
    # for layer in model.layers:
    #     print(layer.name, layer.input_shape, layer.output_shape)

    return model


def SGDNet(basemodel = 'resnet', saliency = 'output', CA = True, img_rows=480, img_cols=640, fixed =False,out1dim=512,out2dim=512, train_bn=None):

    inputimage = Input(shape=(img_rows, img_cols,3))
    
#    input_ml_net2 = Input(shape=(int(img_rows/8), int(img_cols/8),1))
    if basemodel == 'resnet':
        base_model = kerasResnet(input_tensor=inputimage, train_bn=None)
    else: 
        base_model = M_VGG16(input_tensor=inputimage)

    if fixed:
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output

    fraw = Convolution2D(out1dim, kernel_size=(1, 1), activation='relu',padding='same',use_bias=False)(x)
    if CA:
        fca,_= se_block(fraw, compress_rate = 4)   #this is right
    else:
        fca  = fraw     
    #x = BatchNormalization(axis=-1, scale=False)(x)
    if saliency == 'output':
        sm = Convolution2D(1, kernel_size=(1, 1), activation='relu',padding='same',use_bias=False,name='saliency')(fraw)
        fm=layers.multiply([fca,sm])
    elif saliency == 'input': 
        sm = Input(shape=(None, None ,1))
        fm=layers.multiply([fca,sm])
    else:
        fm = fca

    fv= GlobalAveragePooling2D()(fm)

    x = Dense(out2dim, activation='relu', name='fc1')(fv)
    x = Dropout(0.5)(x)
    x = Dense(out2dim, activation='relu', name='fc2')(x)
#    x = Dropout(0.5)(x)
    final_output= Dense(1,name='predictions')(x)
#    , activation='softmax', name='predictions'
#   
#    
    if saliency == 'output':
        model = Model(inputs=[inputimage], outputs=[final_output,sm])  
    elif saliency == 'input': 
        model = Model(inputs=[inputimage,sm], outputs=[final_output])  
    else: 
        model = Model(inputs=[inputimage], outputs=[final_output])
    # for layer in model.layers:
    #     print(layer.name, layer.input_shape, layer.output_shape)

    return model

