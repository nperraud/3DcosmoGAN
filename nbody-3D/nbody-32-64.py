import sys
sys.path.insert(0, '../')
import os
import tensorflow as tf
from gantools import utils
from cosmotools.data import load
from cosmotools.data import fmap
from gantools.model import UpscalePatchWGAN
from cosmotools.model import CosmoWGAN
from gantools.gansystem import GANsystem, UpscaleGANsystem
from functools import partial


shift = 1
c = 20000
forward = partial(fmap.stat_forward, shift=shift, c=c)
backward = partial(fmap.stat_backward, shift=shift, c=c)

ns = 32
try_resume = True

time_str = '32_to_64'
global_path = '../saved_results/nbody/'
name = 'WGAN_' + time_str

bn=False


md=32
params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 1, 2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, md, md]
params_discriminator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn ]
params_discriminator['full'] = [256]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 3
params_discriminator['inception'] = False
params_discriminator['spectral_norm'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1, 1]
params_generator['latent_dim'] = 1024 + 16*16*16
params_generator['latent_dim_split'] = [16,16,16,1]
params_generator['nfilter'] = [md, md, 4*md, 2*md, 1]
params_generator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['data_size'] = 3
params_generator['inception'] = False
params_generator['spectral_norm'] = True
params_generator['use_Xdown'] = True
params_generator['borders'] = dict()
params_generator['borders']['stride'] = [2, 2, 2]
params_generator['borders']['nfilter'] = [md, md, 16]
params_generator['borders']['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_generator['borders']['batch_norm'] = [bn, bn, bn]
params_generator['borders']['data_size'] = 3
params_generator['borders']['width_full'] = None

params_optimization = dict()
params_optimization['batch_size'] = 8
params_optimization['optimization']['epoch']=100000

params_optimization['n_critic'] = 5

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
params['net']['shape'] = [ns, ns, ns, 8] # Shape of the image
params['net']['loss_type'] = 'wasserstein' # loss ('hinge' or 'wasserstein')
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['upscaling'] = 2

params['optimization'] = params_optimization
params['summary_every'] = 500 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 30
params['Nstats_cubes'] = 30


resume, params = utils.test_resume(try_resume, params)


class CosmoUpscalePatchWGAN(UpscalePatchWGAN, CosmoWGAN):
    pass


wgan = UpscaleGANsystem(CosmoUpscalePatchWGAN, params)

dataset = load.load_nbody_dataset(
    spix=ns,
    scaling=4,
    resolution=256,
    Mpch=350,
    patch=True,
    augmentation=True,
    forward_map=forward,
    is_3d=True)

wgan.train(dataset, resume=resume)
