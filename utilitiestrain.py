from __future__ import division
from importlib.resources import path
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
#import pywt
import random
from PIL import Image

def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.
    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

#shape_r shape_c  H,W

def preprocess_label(paths):
    maps=np.zeros((len(paths), 1))

    for i, path in enumerate(paths):
        label = float(path)
        maps[i] = label
    return maps


def preprocess_imagesandsaliencyforiqa(imgpaths1,simgpaths1,shape_r, shape_c ,crop_h=224,crop_w=224, mirror=False):
    # imgs = np.zeros((len(imgpaths1), shape_r, shape_c, 3))
    imgs =[]
    simgs = []
    # maps=np.zeros((len(mappaths2), 1))
    # i=0
    # cv2.imread() loads images as BGR
    for patha,pathb in zip(imgpaths1,simgpaths1):
        image = cv2.imread(patha[1:],1)
        image = preprocess_image(image)
        simage = cv2.imread(pathb[1:],0)
        simage = np.asarray(simage, np.float32)
        simage /= 255.0  
        # img_h, img_w = simage.shape
        # image = padding(image, shape_r, shape_c, 3)
        # simage = padding(simage, shape_r, shape_c, 1)
        image = cv2.resize(image, (shape_c, shape_r))
        simage = cv2.resize(simage, (shape_c, shape_r))
        img_h, img_w, img_d = image.shape
        
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            simage = cv2.copyMakeBorder(simage, 0, pad_h, 0, 
               pad_w, cv2.BORDER_CONSTANT,
               value=(0.0,))
        else:
            image = image

        img_h, img_w,img_d = image.shape
        h_off = random.randint(0, img_h - crop_h)
        w_off = random.randint(0, img_w - crop_w)
        # print h_off
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(image[h_off : h_off+crop_h, w_off : w_off+crop_w], np.float32)
        simage = simage[h_off : h_off+crop_h, w_off : w_off+crop_w]
        newimg_h = int(int(int(int(int(crop_h/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)
        newimg_w = int(int(int(int(int(crop_w/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)
        #simage = cv2.resize(simage, (newimg_h, newimg_w))  
        simage = cv2.resize(simage, (newimg_w, newimg_h))

        # simage = np.asarray(simage, np.float32)
        # image -= mean
        # image /= 127.5
        # simage /= 255.0 
        simage=np.expand_dims(simage, axis=-1) 
        # print simage.shape
        # print image.shape
        # print patha
        if mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            simage = simage[:, :, ::flip]
#            label = label[:, ::flip]
        
        # newimg_h = int(int(int(int(int(shape_r/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)
        # newimg_w = int(int(int(int(int(shape_c/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)/2.0+0.5)
        # #simage = cv2.resize(simage, (newimg_h, newimg_w))  
        # simage = cv2.resize(simage, (newimg_w, newimg_h))
        imgs.append(image)
        # simage=np.expand_dims(simage, axis=-1) 
        # print simage.shape
        simgs.append(simage)
    imgs=np.array(imgs)
    # print imgs.shape
    simgs=np.array(simgs)
    return imgs,simgs
