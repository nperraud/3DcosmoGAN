import numpy as np
import os
from gantools import utils
from . import path
from . import fmap
from gantools.data import transformation
from gantools.data import *
from functools import partial
from gantools import blocks



def load_samples_raw(ncubes=None, resolution=256, Mpch=70):
    ''' Load 2D or 3D raw images

    Arguments
    ---------
    * ncubes : desired number of cubes (if None: all of them)
    * resolution : [256, 512]
    * Mpch : [70, 350]

    '''
    rootpath = path.root_path()
    input_pattern = '{}_nbody_{}Mpc'.format(resolution, Mpch)
    file_ext = '.h5'
    queue = []
    for file in os.listdir(rootpath):
        if file.endswith(file_ext) and input_pattern in file:
            queue.append(os.path.join(rootpath, file))
            # if len(queue) == 10:
            #     break

    if len(queue) == 0:
        raise LookupError('No file founds, check path and parameters')
    raw_images = []
    for file_path in queue:
        raw_images.append(
            utils.load_hdf5(
                filename=file_path, dataset_name='data', mode='r'))
        if type(raw_images[-1]) is not np.ndarray:
            raise ValueError(
                "Data stored in file {} is not of type np.ndarray".format(
                    file_path))

    raw_images = np.array(raw_images).astype(np.float32)


    if ncubes is None:
        return raw_images
    else:
        if ncubes > len(raw_images):
            raise ValueError("Not enough sample")
        else:
            print('Select {} cubes out of {}.'.format(
                ncubes, len(raw_images)))

        return raw_images[:ncubes]


def load_nbody_dataset(
        ncubes=None,
        resolution=256,
        Mpch=350,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True,
        scaling=1,
        is_3d=False,
        patch=False):

    ''' Load a 2D or a 3D nbody images dataset:

     Arguments
    ---------
    * ncubes : desired number of cubes, if None => all of them (default None)
    * resolution : resolution of the original cube [256, 512] (default 256)
    * Mpch : [70, 350] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * is_3d : load a 3d dataset (default False)
    * patch: experimental feature for patchgan
    '''

    # 1) Load raw images
    images = load_samples_raw(ncubes=ncubes, resolution=resolution, Mpch=Mpch)
    print("images shape = ", images.shape)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)

    if (not is_3d):
        sh = images.shape
        images = images.reshape([sh[0]*sh[1], sh[2], sh[3]])

    # 2p) Apply downscaling if necessary
    if scaling>1:
        if is_3d:
            data_shape = 3
        else:
            data_shape = 2
        images = blocks.downsample(images, scaling, size=data_shape)

    if augmentation:
        # With the current implementation, 3d augmentation is not supported
        # for 2d scaling
        if is_3d:
            t = partial(transformation.random_transformation_3d, roll=True)
        else:
            t = partial(transformation.random_transformation_2d, roll=True)
    else:
        t = None
    
    # 5) Make a dataset
    if patch:
        if is_3d:
            dataset = Dataset_3d_patch(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d_patch(images, spix=spix, shuffle=shuffle, transform=t)

    else:
        if is_3d:
            dataset = Dataset_3d(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d(images, spix=spix, shuffle=shuffle, transform=t)

    return dataset



    
# def load_time_dataset(
#         resolution=256,
#         Mpch=100,
#         shuffle=True,
#         forward_map = None,
#         spix=128,
#         augmentation=True):

#     ''' Load a 2D dataset object 

#      Arguments
#     ---------
#     * resolution : [256, 512] (default 256)
#     * Mpch : [100, 500] (default 70)
#     * shuffle: shuffle the data (default True)
#     * foward : foward mapping use None for raw data (default None)
#     * spix : resolution of the image (default 128)
#     * augmentation : use data augmentation (default True)
#     '''

#     # 1) Load raw images
#     images = load_time_cubes(resolution=resolution, Mpch=Mpch)
#     # (ts, resolution, resolution, resolution)

#     # 2) Apply forward map if necessary
#     if forward_map:
#         images = forward_map(images)
#     if augmentation:
#         t = transformation.random_transformation_3d
#     else:
#         t = None

#     # 5) Make a dataset
#     dataset = Dataset_time(X=images, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

#     return dataset
