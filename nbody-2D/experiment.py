import sys
sys.path.insert(0, '../')

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf

from cosmotools.data import load
from gantools import utils
from gantools.model import WGAN
from gantools.gansystem import GANsystem
from cosmotools.data import fmap
import functools
from copy import deepcopy

from cosmotools.metric import evaluation
from cosmotools.model import CosmoWGAN


if len(sys.argv) > 1:
    ns = int(sys.argv[1])
else:
    ns = 32 #(between 32 and 256)

try_resume = True # Try to resume previous simulation
Mpch = 350 # Type of dataset (select 70 or 350)

forward = fmap.stat_forward
backward = fmap.stat_backward
def non_lin(x):
    return tf.nn.relu(x)

global_path = '../saved_results/nbody-2d/'

dataset = load.load_nbody_dataset(ncubes=30, spix=ns, Mpch=Mpch, forward_map=forward)

name = 'WGAN{}'.format(ns) + 'test_full_' + '2D'

bn = False

md=32

params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 2, 1]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 2*md, md]
params_discriminator['shape'] = [[4, 4],[4, 4],[4, 4], [4, 4], [4, 4]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn ]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['inception'] = False
params_discriminator['spectral_norm'] = False
params_discriminator['fft_features'] = False
params_discriminator['psd_features'] = False

params_generator = dict()
params_generator['stride'] = [1, 2, 2, 2, 1]
params_generator['latent_dim'] = ns*2
params_generator['in_conv_shape'] =[ns//8,ns//8]
params_generator['nfilter'] = [md, 2*md, 4*md, 2*md, 1]
params_generator['shape'] = [[4, 4],[4, 4], [4, 4],[4, 4],[4, 4]]
params_generator['batch_norm'] = [bn, bn, bn,bn ]
params_generator['full'] = [(ns//8)**2 *8]
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['data_size'] = 2
params_generator['inception'] = False
params_generator['spectral_norm'] = False


params_optimization = dict()
params_optimization['batch_size'] = 32
params_optimization['epoch'] = (ns**2)//64
params_optimization['n_critic'] = 5
# params_optimization['generator'] = dict()
# params_optimization['generator']['optimizer'] = 'adam'
# params_optimization['generator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['generator']['learning_rate'] = 0.0004
# params_optimization['discriminator'] = dict()
# params_optimization['discriminator']['optimizer'] = 'adam'
# params_optimization['discriminator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['discriminator']['learning_rate'] = 0.0001

# Cosmology parameters
params_cosmology = dict()
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward


# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['cosmology'] = params_cosmology # Parameters for the cosmological summaries
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [ns, ns, 1] # Shape of the image
params['net']['loss_type'] = 'wasserstein' # loss ('hinge' or 'wasserstein')
params['net']['gamma_gp'] = 10 # Gradient penalty

params['optimization'] = params_optimization
params['summary_every'] = 500 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 2000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = (64*32*32)//ns



resume, params = utils.test_resume(try_resume, params)
# If a model is reloaded and some parameters have to be changed, then it should be done here.
# For example, setting the number of epoch to 5 would be:

wgan = GANsystem(CosmoWGAN, params)


wgan.train(dataset, resume=resume)
