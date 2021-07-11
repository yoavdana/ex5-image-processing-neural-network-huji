

 



import re
import os,itertools, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.draw import line

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import color




def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:    
        im = im.astype(np.float64) / 255.0
    return im
########## End of utils ##########

########## Download datasets ##########



########## End of download datasets ##########

"""# 3 Dataset Handling"""

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    LEN_FILE=len(filenames)
    images_dict={}
    crop_row,crop_col=crop_size[0],crop_size[1]
    while True:
        source_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        target_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        for i in range(batch_size):
            rand_idx = random.randint(0,LEN_FILE-1)
            if filenames[rand_idx] not in images_dict:
                images_dict[filenames[rand_idx]]=read_image(filenames[rand_idx],1)      
            curr_im=images_dict[filenames[rand_idx]]
            rand_row, rand_col=rand_slice(3*crop_col,3*crop_row,curr_im) 
            sliced_im = curr_im[rand_row:rand_row + 3*crop_row,rand_col:rand_col + 3*crop_col]
            corrupt=corruption_func(sliced_im)
            rand_row, rand_col=rand_slice(crop_col,crop_row,sliced_im)
            sliced_im =sliced_im[rand_row:rand_row + crop_row,rand_col:rand_col + crop_col]-0.5
            sliced_corrupt=corrupt[rand_row:rand_row + crop_row,rand_col:rand_col + crop_col]-0.5
            source_batch[i, :, :, :]=sliced_corrupt.reshape(crop_row,crop_col,1)
            target_batch[i, :, :, :]=sliced_im.reshape(crop_row,crop_col,1)
        yield (source_batch,target_batch)

def rand_slice(crop_col, crop_row,curr_im):
    rand_row, rand_col = random.randint(0, curr_im.shape[0]-crop_row),random.randint(0,curr_im.shape[1]-crop_col)
    return rand_row, rand_col

"""# 4 Neural Network Model"""

def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    X=Conv2D(num_channels,(3,3),padding='same')(input_tensor)
    X=Activation ('relu')(X)
    O = Conv2D(num_channels, (3,3), padding='same')(X)
    addi_=Add ()([input_tensor,O])
    out= Activation('relu')(addi_)
    return out

def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    input=Input(shape =(height,width,1))
    input_conv=Conv2D(num_channels,(3,3),padding='same')(input)
    input_to_res=Activation('relu')(input_conv)
    for res in range(num_res_blocks):
        input_to_res=resblock(input_to_res, num_channels)
    out_conv=Conv2D(1,(3,3),padding='same')(input_to_res)
    output=Add ()([ input,out_conv])
    return Model(inputs=input,outputs=output)

"""# 5 Training Networks for Image Restoration"""

def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    split_ind=int(len(images)*0.8)
    trannig_set,validation_set=images[:split_ind],images[split_ind:]
    crop_size=model.input_shape[1:3]
    train_data_set=load_dataset(trannig_set, batch_size, corruption_func, crop_size)
    valid_data_set=load_dataset(validation_set, batch_size, corruption_func,crop_size)
    model.compile(loss='mean_squared_error',optimizer=Adam(beta_2=0.9))
    Fit=model.fit_generator(train_data_set,steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,validation_data=valid_data_set,validation_steps=num_valid_samples/batch_size)

"""# 6 Image Restoration of Complete Images"""

def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    #set new model
    height,width=corrupted_image.shape
    in_=Input(shape=(height,width,1))
    _out=base_model(in_)
    new_model=Model(inputs=in_,outputs=_out)
    new_model.set_weights(base_model.get_weights())
    #set image for the process
    pre_pro_im=corrupted_image-0.5
    pre_pro_im =pre_pro_im[np.newaxis,...,np.newaxis]
    #restore image
    restored_image=new_model.predict(pre_pro_im)[0]
    restored_image +=0.5
    restored_image=np.clip(restored_image,0,1)
    return restored_image

"""# 7 Application to Image Denoising and Deblurring
## 7.1 Image Denoising
### 7.1.1 Gaussian Noise
"""

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sigma=np.random.uniform(min_sigma,max_sigma)
    ga_noise=np.random.normal(0,sigma,image.shape)
    noise_im=(np.round((image+ga_noise)*255)/255).clip(0,1)
    return  noise_im

#@markdown ### 7.1.2 Training a Denoising Mode

denoise_num_res_blocks = 7 #@param {type:"slider", min:1, max:15, step:1}
#@markdown **DON'T FORGET TO RUN THIS CELL AFTER CHANGING IT!**

def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode==False:
        batch_size=100
        steps_per_epoch=100
        num_epochs=10
        num_valid_samples=1000
    else:
        batch_size=10
        steps_per_epoch=3
        num_epochs=2
        num_valid_samples=30
    num_channels = 48
    patch_size = 24
    images=images_for_denoising()
    corruption_func= lambda img: add_gaussian_noise(img, 0,0.2)
    model=build_nn_model(patch_size,patch_size, num_channels,denoise_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch,num_epochs, num_valid_samples)
    return model

"""## 7.2 Image Deblurring
### 7.2.1 Motion Blur
"""

def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    filter=motion_blur_kernel(kernel_size, angle)
    blured_im=convolve(image,filter)
    return blured_im

def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle=np.random.uniform(0,np.pi)
    ind=np.random.randint(len(list_of_kernel_sizes))
    kernel_size=list_of_kernel_sizes[ind]
    blured=add_motion_blur(image,kernel_size,angle)
    corupt = (np.round(blured * 255) / 255).clip(0, 1)
    return corupt

#@markdown ### 7.2.2 Training a Deblurring Model


deblur_num_res_blocks = 7 #@param {type:"slider", min:1, max:15, step:1}

#@markdown **DON'T FORGET TO RUN THIS CELL AFTER YOU CHANGING THE VALUE!**

def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode==False:
        batch_size=100
        steps_per_epoch=100
        num_epochs=10
        num_valid_samples=1000
    else:
        batch_size=10
        steps_per_epoch=3
        num_epochs=2
        num_valid_samples=30
    num_channels = 32
    patch_size = 16
    images=images_for_deblurring()
    kernel_size=[7]
    corruption_func= lambda img: random_motion_blur(img,kernel_size)
    model=build_nn_model(patch_size,patch_size, num_channels,deblur_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch,num_epochs, num_valid_samples)
    return model

"""##7.3 Image Super-resolution
### 7.3.1 Image Low-Resolution Corruption
**Note:** Make sure your implementation covers different LR scales, for simplicity you may assume we won't test your network on images which are of more than $4$ times lower in quality compared to the HR one.

**Hint:** The `scipy.ndimage.zoom` function may come in handy.

"""

from scipy.ndimage import map_coordinates
def super_resolution_corruption(image):
    """
    Perform the super resolution corruption 
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    min_change=2
    max_change=4
    height,width=image.shape
    res_change=np.random.randint(min_change,max_change)
    #nearest_rational=np.round(res_change*max_change)/max_change
    shrink_factor=1/res_change
    x, y = np.indices((height,width)).astype(np.float64)
    pre_im_n=zoom(image,shrink_factor)
    corupt = zoom(pre_im_n, res_change)
    corupt_n=map_coordinates(corupt, [x.flatten(), y.flatten()], order=1,
                     prefilter=False).reshape(image.shape)
    return corupt_n

#@markdown ### 7.3.2 Training a Super Resolution Model


super_resolution_num_res_blocks = 7 #@param {type:"slider", min:1, max:15, step:1}
batch_size = 64 #@param {type:"slider", min:1, max:128, step:16}
steps_per_epoch = 2500 #@param {type:"slider", min:100, max:5000, step:100}
num_epochs = 10 #@param {type:"slider", min:1, max:20, step:1}
patch_size = 32 #@param {type:"slider", min:8, max:32, step:2}
num_channels = 48 #@param {type:"slider", min:16, max:64, step:2}

#@markdown **DON'T FORGET TO RUN THIS CELL AFTER YOU CHANGING THE VALUES!**

def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode == False:
        batch_size = 100
        steps_per_epoch = 200
        num_epochs = 15
        num_valid_samples = 1000
    else:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30
    num_channels =60
    patch_size = 30
    images = images_for_super_resolution()
    corruption_func = lambda img: super_resolution_corruption(img)
    model = build_nn_model(patch_size, patch_size, num_channels,
                           super_resolution_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch,
                num_epochs, num_valid_samples)
    return model






