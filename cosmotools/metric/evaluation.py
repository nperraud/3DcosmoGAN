"""Evaluation module.

This module contains helping functions for the evaluation of the models.
"""

import tensorflow as tf
import pickle
import numpy as np
from . import stats
from gantools.plot import plot_with_shade
import matplotlib
import socket
if 'nid' in socket.gethostname() or 'lo-' in socket.gethostname():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from gantools.model import *
from gantools.gansystem import *
from .score import safe_fd,fd2score, lim_hist, lim_peak


def generate_samples(obj, N=None, checkpoint=None, **kwards):
    """Generate sample from gan object."""
    gen_sample, gen_sample_raw = obj.generate(
        N=N, checkpoint=checkpoint, **kwards)
    gen_sample = np.squeeze(gen_sample)
    gen_sample_raw = np.squeeze(gen_sample_raw)
    return gen_sample, gen_sample_raw


# Compute and plot PSD from raw images
# multiply is a boolean flag indicating whether to multiply the PSD by k*(k+1)/(2*pi)
# box_l indicates the image resolution in radians
# bin_k is the number of bins
# confidence is either None, a number between 0 and 1 or 'std'. If 'std' then the standard deviation is plotted as shading, otherwise either the confidence interval or nothing
# fractional_difference: if true the fractional difference is plotted as well
# log_sampling: whether the bins are logarithmically sampled
# cut: either None or an interval indicating where to cut the PSD
def compute_and_plot_psd(raw_images, gen_sample_raw, multiply=False, box_ll=350, bin_k=50, confidence=None, ylim=None, fractional_difference=False, log_sampling=True, cut=None, display=True, ax=None, loc=1, lenstools=False, **kwargs):
    
    # Compute PSD
    if lenstools:
        raise NotImplementedError('Need to be fixed')
        psd_real, x = stats.psd_lenstools(raw_images, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply)
        psd_gen, x = stats.psd_lenstools(gen_sample_raw, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply)
    else:
        psd_real, x = stats.power_spectrum_batch_phys(X1=raw_images, multiply=multiply, bin_k=bin_k, box_ll=box_ll, log_sampling=log_sampling, cut=cut)
        psd_gen, x = stats.power_spectrum_batch_phys(X1=gen_sample_raw, multiply=multiply, bin_k=bin_k, box_ll=box_ll, log_sampling=log_sampling, cut=cut)
    # Compute the mean
    psd_real_mean = np.mean(psd_real, axis=0)
    psd_gen_mean = np.mean(psd_gen, axis=0)

#     # Compute the difference between the curves
#     l2, logel2, l1, logel1 = stats.diff_vec(psd_real_mean, psd_gen_mean)
#     frac_diff = stats.fractional_diff(psd_real_mean, psd_gen_mean).mean()
    npix = np.prod(raw_images.shape[1:])
    d = safe_fd(psd_real,psd_gen, npix)
    score = fd2score(d)
    if display:
        print('PSD Frechet Distance: {}\n'
              'PSD Score           : {}\n'.format(d, score))
#         print('Log l2 PSD loss: {}\n'
#               'L2 PSD loss: {}\n'
#               'Log l1 PSD loss: {}\n'
#               'L1 PSD loss: {}\n'
#               'Fractional difference: {}'.format(logel2, l2, logel1, l1, frac_diff))
    
    # Plot the two curves
    plot_cmp(x, psd_gen, psd_real, ax=ax, xscale='log', yscale='log', xlabel='$k$' if multiply else '$l$', ylabel='$\\frac{l(l+1)P(l)}{2\pi}$' if multiply else '$P(k)$', title="Power spectral density", shade=True, confidence=confidence, ylim=ylim, fractional_difference=fractional_difference, loc=loc)
    return score

def compute_and_plot_peak_count(raw_images, gen_sample_raw, display=True, ax=None, log=True, lim = lim_peak, neighborhood_size=5, threshold=0, confidence=None, ylim=None, fractional_difference=False, algo='relative', loc=1, **kwargs):
    """Compute and plot peak count histogram from raw images."""
    y_real, y_fake, x = stats.peak_count_hist_real_fake(raw_images, gen_sample_raw, log=log, lim=lim, neighborhood_size=neighborhood_size, threshold=threshold, mean=False)
#     l2, logel2, l1, logel1 = stats.diff_vec(np.mean(y_real, axis=0), np.mean(y_fake, axis=0))
#     rel_diff = None
#     if confidence is not None:
#         rel_diff = stats.relative_diff(y_real, y_fake).mean()
#     if display:
#         print('Log l2 Peak Count loss: {}\n'
#               'L2 Peak Count loss: {}\n'
#               'Log l1 Peak Count loss: {}\n'
#               'L1 Peak Count loss: {}'.format(logel2, l2, logel1, l1))
    npix = np.prod(raw_images.shape[1:])
    d = safe_fd(y_real,y_fake, npix)
    score = fd2score(d)
    if display:
        print('Peak Frechet Distance: {}\n'
              'Peak Score           : {}\n'.format(d, score))

    plot_cmp(x, y_fake, y_real, title= 'Peak histogram', xlabel='Size of the peaks', ylabel='Pixel count', ax=ax, xscale='log' if log else 'linear', shade=True, confidence=confidence, ylim=ylim, fractional_difference=fractional_difference, algorithm=algo, loc=loc)
    return score


def compute_and_plot_mass_hist(raw_images, gen_sample_raw, display=True, ax=None, log=True, lim=lim_hist, confidence=None, ylim=None, fractional_difference=False, algo='relative', loc=1, **kwargs):
    """Compute and plot mass histogram from raw images."""
#     raw_max = 250884    
#     lim = [np.log10(1), np.log10(raw_max/3)]
    y_real, y_fake, x = stats.mass_hist_real_fake(raw_images, gen_sample_raw, log=log, lim=lim, mean=False)
#     l2, logel2, l1, logel1 = stats.diff_vec(np.mean(y_real, axis=0), np.mean(y_fake, axis=0))
#     rel_diff = None
#     if confidence is not None:
#         rel_diff = stats.relative_diff(y_real, y_fake).mean()
#     if display:
#         print('Log l2 Mass histogram loss: {}\n'
#               'L2 Peak Mass histogram: {}\n'
#               'Log l1 Mass histogram loss: {}\n'
#               'L1 Mass histogram loss: {}'.format(logel2, l2, logel1, l1))
    npix = np.prod(raw_images.shape[1:])
    d = safe_fd(y_real,y_fake, npix)
    score = fd2score(d)
    if display:
        print('Mass Frechet Distance: {}\n'
              'Mass Score           : {}\n'.format(d, score))
        
    plot_cmp(x, y_fake, y_real, title='Mass histogram', xlabel='Number of particles', ylabel='Pixel count', ax=ax, xscale='log' if log else 'linear', shade=True, confidence=confidence, ylim=ylim, fractional_difference=fractional_difference, algorithm=algo, loc=loc)
    return score


# Compute same histogram as in mustafa (Figure 3b)
def compute_plot_psd_mode_hists(raw_images, gen_sample_raw, modes=1, multiply=True, box_l=(5*np.pi/180), bin_k=50, log_sampling=False, cut=None, hist_bin=20, hist_batch=1, confidence=None, lenstools=False, loc=1):
    
    # Compute PSD    
    if lenstools:
        psd_real, x = stats.psd_lenstools(raw_images, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply)
        psd_gen, x = stats.psd_lenstools(gen_sample_raw, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply)
    else:
        psd_real, x = stats.power_spectrum_batch_phys(X1=raw_images, multiply=multiply, bin_k=bin_k, box_l=box_l, log_sampling=log_sampling, cut=cut)
        psd_gen, x = stats.power_spectrum_batch_phys(X1=gen_sample_raw, multiply=multiply, bin_k=bin_k, box_l=box_l, log_sampling=log_sampling, cut=cut)
    
    # Get modes
    idx = np.linspace(0, len(x) - 1, modes + 2).astype(np.int)[1:-1]

    # Compute histograms
    vmins = np.min(np.vstack([np.min(psd_real[:, idx], axis=0), np.min(psd_gen[:, idx], axis=0)]), axis=0)
    vmaxs = np.max(np.vstack([np.max(psd_real[:, idx], axis=0), np.max(psd_gen[:, idx], axis=0)]), axis=0)
    batch_size = len(psd_real) // hist_batch
    def compute_hist(psd, mode):
        histo = []
        for i in range(hist_batch):
            h, e = np.histogram(psd[i*batch_size:(i+1)*batch_size, idx[mode]], bins=hist_bin, range=(vmins[mode], vmaxs[mode]))
            histo.append(h)
        return np.array(histo), (e[:-1] + e[1:]) / 2
    
    real_hists = [compute_hist(psd_real, i) for i in range(modes)]
    fake_hists = [compute_hist(psd_gen, i) for i in range(modes)]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=modes, figsize=(5 * modes, 5))
    if modes == 1:
        ax = [ax]
    for i in range(modes):
        compute_and_plot_peak_cout(real_hists[i][1], fake_hists[i][0], real_hists[i][0], ax=ax[i], xlabel='$\\frac{l(l+1)P(l)}{2\pi}$' if multiply else '$P(l)$', ylabel='Pixel count', title="$l=" + str(int(x[idx[i]])) + "$", confidence=confidence, shade=True, loc=1)
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.tight_layout()

    
# def loglog_cmp_plot(x, y_real, y_fake, title='', xlabel='', ylabel='', persample=None):
#     if persample is None:
#         persample = len(y_real.shape)>1
    
#     fig = plt.figure(figsize=(5,3))
#     ax = plt.gca()
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     linestyle = {
#         "linewidth": 1,
#         "markeredgewidth": 0,
#         "markersize": 3,
#         "marker": "o",
#         "linestyle": "-"
#     }
#     if persample:
#         plot.plot_with_shade(
#             ax,
#             x,
#             y_fake,
#             color='r',
#             label="Fake",
#             **linestyle)
#         plot.plot_with_shade(
#             ax,
#             x,
#             y_real,
#             color='b',
#             label="Real",
#             **linestyle)
#     else:
#         ax.plot(x, y_fake, label="Fake", color='r', **linestyle)
#         ax.plot(x, y_real, label="Real", color='b', **linestyle)

#     # ax.set_ylim(bottom=0.1)
#     ax.title.set_text(title)
#     ax.title.set_fontsize(16)
#     ax.set_xlabel(xlabel, fontsize=14)
#     ax.set_ylabel(ylabel, fontsize=14)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.legend(fontsize=14, loc=3)
#     return ax





# ------------------------------------------------------------------------------------------ #
# Everything below this line still needs to be cleaned and will be used in a future version  #
# ------------------------------------------------------------------------------------------ #

def equalizing_histogram(real, fake, nbins=1000, bs=2000):
    sh = fake.shape
    real = real.flatten()
    fake = fake.flatten()
    v, x = np.histogram(real, bins=nbins)
    v2, x2 = np.histogram(fake, bins=nbins)
    c = np.cumsum(v) / np.sum(v)
    c2 = np.cumsum(v2) / np.sum(v2)
    ax = np.cumsum(np.diff(x)) + np.min(x)
    ax2 = np.cumsum(np.diff(x2)) + np.min(x2)
    N = len(fake)
    res = []
    print('This implementation is slow...')
    for index, batch in enumerate(np.split(fake, N // bs)):
        if np.mod(index, N // bs // 100) == 0:
            print('{}% done'.format(bs * index * 100 // N))
        ind1 = np.argmin(np.abs(np.expand_dims(ax2, axis=0) - np.expand_dims(batch, axis=1)), axis=1)
        ind2 = np.argmin(np.abs(np.expand_dims(c, axis=0) - np.expand_dims(c2[ind1], axis=1)), axis=1)
        res.append(ax[ind2])
    return np.reshape(np.concatenate(res), sh)


# Plot the statistics in the given row of the subplot for the given fake and real samples
# fake and real can eithe be an array of samples, None (in case of real images) or a function to load the array of samples
def plot_stats(row, fake, real=None, confidence='std', box_l=(5*np.pi/180), bin_k=50, multiply=True, cut=None, ylims=[None, None, None], fractional_difference=[False, False, False], locs=[2, 1, 1], lenstools=False, **kwargs):
    stat = 0
    scores = []

    # Load the data if fake and real are functions
    if callable(fake):
        fake = fake()
    if callable(real):
        real = real()

    # Compute the statistics and plot
    for col in row:
        if real is None:

            # Produce plots for fake only
            if stat == 0:
                # Power spectral density
                if lenstools:
                    psd_gen, x = stats.psd_lenstools(fake, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply)
                else:
                    psd_gen, x = stats.power_spectrum_batch_phys(X1=fake, box_l=box_l, bin_k=bin_k, cut=cut, multiply=multiply, log_sampling=False)
                plot_cmp(x, psd_gen, ax=col, xscale='log', xlabel='$l$', ylabel='$\\frac{l(l+1)P(l)}{2\pi}$' if multiply else '$P(l)$', title="Power spectral density", shade=True, confidence=confidence, ylim=ylims[stat], fractional_difference=fractional_difference[stat], loc=locs[stat])
                # plot_single(x, psd_gen, color='r', ax=col, xlabel='$l$', ylabel='$\\frac{l(l+1)P(l)}{2\pi}$'if multiply else '$P(l)$', title="Power spectral density", xscale='log', shade=True, confidence=confidence, ylim=ylims[stat])
            elif stat == 1:
                # Peak count
                y, x, _ = stats.peak_count_hist(fake, mean=False, **kwargs)
                plot_cmp(x, y, title='Peak histogram', xlabel='Size of the peaks', ylabel='Pixel count', ax=col, shade=True, confidence=confidence, ylim=ylims[stat], fractional_difference=fractional_difference[stat], loc=locs[stat], algorithm='relative')
                # plot_single(x, y, color='r', ax=col, title='Peak histogram', xlabel='Size of the peaks', ylabel='Pixel count', xscale='linear', shade=True, confidence=confidence, ylim=ylims[stat])
            else:
                # Mass histogram
                y, x, _ = stats.mass_hist(fake, mean=False, **kwargs)
                plot_cmp(x, y, title='Mass histogram', xlabel='Number of particles', ylabel='Pixel count', ax=col, shade=True, confidence=confidence, ylim=ylims[stat], fractional_difference=fractional_difference[stat], loc=locs[stat], algorithm='relative')
                # plot_single(x, y, color='r', ax=col, title='Mass histogram', xlabel='Number of particles', ylabel='Pixel count', xscale='linear', shade=True, confidence=confidence, ylim=ylims[stat])
        else:    
        
            # Produce plots for both real and fake
            if stat == 0:
                s = compute_and_plot_psd(real, fake, ax=col, display=False, confidence=confidence, multiply=multiply, ylim=ylims[stat], box_l=box_l, bin_k=bin_k, cut=cut, fractional_difference=fractional_difference[stat], loc=locs[stat], lenstools=lenstools, **kwargs)
            elif stat == 1:
                s = compute_and_plot_peak_count(real, fake, ax=col, display=False, confidence=confidence, ylim=ylims[stat], fractional_difference=fractional_difference[stat], loc=locs[stat], **kwargs)
            else:
                s = compute_and_plot_mass_hist(real, fake, ax=col, display=False, confidence=confidence, ylim=ylims[stat], fractional_difference=fractional_difference[stat], loc=locs[stat], **kwargs)
            scores.append(s)
        stat = stat + 1
    return scores

# Compute all the statistic plots for a set of parameters (in 1D)
# Produces n_params * 3 plots, where every row corresponds to a set of parameters
# and every column to a different statistic
def compute_plots_for_params(params, real, fake, param_str=(lambda x: str(x[0])[0:7]), **kwargs):
    fig, ax = plt.subplots(nrows=len(params), ncols=3, figsize=(15, 5 * len(params)))
    idx = 0
    score = []
    for row in ax:
        stat = 0
        s = plot_stats(row, fake[idx], real[idx], **kwargs)
        if idx > 0:
            for col in row:
                col.set_title('')
        if idx < len(params) - 1:
            for col in row:
                col.set_xlabel('')
        idx = idx + 1
        score.append(s)
    for a, param in zip(ax[:,2], params):
        ar = a.twinx()
        ar.set_ylabel(param_str(param), labelpad=50, fontsize=14)
        ar.set_yticks([])
    fig.tight_layout()
    return fig, np.array(score)


# Returns the frames to create a video with moviepy
# Each frame consists of the three statistic functions, a real and a fake image
# X is a list of dictionary containing object with the following signature:
# {'fake': fake images, 'real': real images (or None), 'params': parameters}
# if save_frames_folder is set, frames are saved into that folder
# returns a list of frames
def make_frames(X, title_func=(lambda x: x), display_loss=False, vmin=None, vmax=None, params_grid=None, transform=(lambda x: x), save_frames_dir=None, dpi=150, **kwargs):

    if save_frames_dir is not None and not os.path.exists(save_frames_dir):
        os.makedirs(save_frames_dir)

    # Precompute frames
    frames = []
    for i in range(len(X)):
        
        dic = X[i]

        # Load images if they were not loaded
        if callable(dic['fake']):
            fake = dic['fake']()
        else:
            fake = dic['fake']
        if callable(dic['real']):
            real = dic['real']()
        else:
            real = dic['real']

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        row_idx = 0
        for row in ax:
            if row_idx == 0:
                stats_txt = plot_stats(row, fake, real, **kwargs)
            else:
                col_idx = 0
                for col in row:
                    col.axis('off')
                    if col_idx == 0:
                        col.set_title("Fake")
                        plot_img(transform(fake[np.random.randint(0, len(fake))]), vmin=transform(vmin), vmax=transform(vmax), ax=col)
                    elif col_idx == 1 and real is not None:
                        col.set_title("Real")
                        plot_img(transform(real[np.random.randint(0, len(real))]), vmin=transform(vmin), vmax=transform(vmax), ax=col)
                    elif col_idx == 2 and params_grid is not None:
                        col.scatter(params_grid[:, 0], params_grid[:, 1])
                        col.scatter(dic['params'][0], dic['params'][1], color='r', s=80)
                        col.set_xlabel('$\Omega_M$')
                        col.set_ylabel('$\sigma_8$')
                        col.axis('on')
                    elif display_loss and stats_txt != '':
                        col.text(0.5, 0.5, stats_txt, ha='center', va='center')
                    col.title.set_fontsize(16)
                    col_idx = col_idx + 1
            row_idx = row_idx + 1
        fig.tight_layout()
        plt.figtext(0.5, 0.01, '\n' + title_func(dic['params']), wrap=True, horizontalalignment='center', fontsize=20)
        if save_frames_dir is not None:
            plt.savefig(os.path.join(save_frames_dir, 'frame_' + str(i) + '.png'), dpi=dpi)
        plt.close()
        fig = mplfig_to_npimage(fig)
        frames.append(fig)

        # Save memory
        del real
        del fake
    return frames


# Returns the make_frame function to create a video with moviepy
# X is a list of dictionary containing object with the following signature:
# {'fake': fake images, 'real': real images (or None), 'params': parameters}
# frames is a list of frames or a folder containing the frames
# duration indicates the duraion of each frame (in seconds)
# frames_stat indicates for how many frames comparisons between real and fake should stay
def make_frame_func(X, frames, duration, frames_stat=1):

    # if frames is a folder load images from that folder
    if type(frames) is str:
        new_frames = []
        for i in range(len(os.listdir(frames))):
            img = mpimg.imread(os.path.join(frames, 'frame_' + str(i) + '.png'))
            img = img[:, :, :3] * 255
            new_frames.append(img.astype('uint8'))
        frames = new_frames

    # Expand frames
    new_frames = []
    for i in range(len(X)):
        for f in range(1 if X[i]['real'] is None else frames_stat):
            new_frames.append(frames[i])

    # Define make_frame function
    def make_frame(t):
        t = int((len(new_frames) / duration) * t)
        return new_frames[t]

    return make_frame


def generate_samples_params(wgan, params, nsamples=1, checkpoint=None):
    if params.ndim == 1:
        p = np.tile(params, [nsamples, 1])
    else:
        p = params
    latent = wgan.net.sample_latent(bs=nsamples, params=p)
    return wgan.generate(N=nsamples, **{'z': latent}, checkpoint=checkpoint)


def generate_samples_same_seed(wgan, params, nsamples=1, checkpoint=None):

    # Sample latent variables
    init_params = np.array([[wgan.net.params['init_range'][c][0] for c in range(wgan.net.params['cond_params'])]])
    latent_0 = wgan.net.sample_latent(bs=nsamples, params=init_params)
    z_0 = latent_0.reshape((nsamples, wgan.net.params['generator']['latent_dim'], wgan.net.params['cond_params']))
    z_0 = z_0 / wgan.net.params['final_range'][0]

    # Scale parameters to range
    gen_params = np.zeros((len(params), wgan.net.params['cond_params']))
    for i in range(len(gen_params)):
        for c in range(wgan.net.params['cond_params']):
            gen_params[i, c] = utils.scale2range(params[i][c], wgan.net.params['init_range'][c], wgan.net.params['final_range'])

    # Normalise the distribution to the final range
    latents = []
    for idx in range(len(gen_params)):
        z = np.copy(z_0)
        for c in range(wgan.net.params['cond_params']):
            z[:, :, c] = z[:, :, c] * gen_params[idx, c]
        latents.append(z.reshape((nsamples, wgan.net.params['cond_params'] * wgan.net.params['generator']['latent_dim'])))

    # Generate images
    gen_imgs = []
    for latent in latents:
        gen_imgs.append(wgan.generate(N=nsamples, **{'z': latent}, checkpoint=checkpoint))
    return gen_imgs


# Produce and save generated images
# Returns dataset file
def generate_save(filename, wgan, params, N=100, checkpoint=None):
    file_name = filename + str(checkpoint) + '.h5'
    first = True
    for p in params:
        gen_params = np.tile(p, [N, 1])
        gen_imgs = generate_samples_params(wgan, p, nsamples=N, checkpoint=checkpoint)
        utils.append_h5(file_name, gen_imgs, gen_params, overwrite=first)
        first = False
    return load_params_dataset(filename=file_name, batch=N, sorted=True)


# Compute and plot the PSD correlation matrices of real and fake samples
# box_l is the resolution of the images in sky degrees (radians)
# bin_k is the number of bins
# log_sampling is a flag that indicates whether the bins should be logarithmically samples or not
# cut is either None or in the format [a, b], where all the frequencies below a and above b are cut
# the real correlation matrix, the fake correlation matrix and the axis k are returned
def compute_plot_correlation(real, fake, tick_every=3, box_l=(5*np.pi/180), bin_k=50, ax=None, log_sampling=False, cut=None, to_plot=True, lenstools=False):

    # Compute the correlations
    if lenstools:
        corr_real, k = stats.psd_correlation_lenstools(real, box_l=box_l, bin_k=bin_k, cut=cut)
        corr_fake, _ = stats.psd_correlation_lenstools(fake, box_l=box_l, bin_k=bin_k, cut=cut)
    else:
        corr_real, k = stats.psd_correlation(real, bin_k=bin_k, box_l=box_l, log_sampling=log_sampling, cut=cut)
        corr_fake, _ = stats.psd_correlation(fake, bin_k=bin_k, box_l=box_l, log_sampling=log_sampling, cut=cut)

    # Plot
    if to_plot:
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        im = plot_img(np.abs(corr_real), x=k, ax=ax[0], title='Real', vmin=0, vmax=1, tick_every=tick_every)
        plot_img(np.abs(corr_fake), x=k, ax=ax[1], title='Fake', vmin=0, vmax=1, tick_every=tick_every)
        plot_img(np.abs(corr_fake - corr_real), x=k, ax=ax[2], title='abs(Real - Fake)', vmin=0, vmax=1, tick_every=tick_every)
        plt.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    return corr_real, corr_fake, k


def compute_correlations(real_images, fake_images, params, box_l=(5*np.pi/180), bin_k=50, cut=None, log_sampling=False, lenstools=False):

    # Local function to load a subset of images
    def load_imgs(X):
        if callable(X):
            return X()
        return X

    corr = []
    for idx in range(len(params)):
        real = load_imgs(real_images[idx])
        fake = load_imgs(fake_images[idx])
        c_r, c_f, k = compute_plot_correlation(real, fake, box_l=box_l, bin_k=bin_k, log_sampling=log_sampling, cut=cut, to_plot=False, lenstools=lenstools)
        corr.append([c_r, c_f])
    return corr, k


def plot_correlations(corr, k, params, param_str=(lambda x: str(x)), tick_every=10, figsize=None):

    fig, ax = plt.subplots(nrows=len(params), ncols=3, figsize=(15, len(params) * 5.5) if figsize is None else figsize)
    idx = 0
    scores = []
    for row in ax:
        c_r, c_f = corr[idx]
        im = plot_img(np.abs(c_r), x=k, ax=row[0], vmin=0, vmax=1, tick_every=tick_every)
        plot_img(np.abs(c_f), x=k, ax=row[1], vmin=0, vmax=1, tick_every=tick_every)
        plot_img(np.abs(c_r - c_f), x=k, ax=row[2], vmin=0, vmax=1, tick_every=tick_every)
        plt.colorbar(im, ax=row.ravel().tolist(), shrink=0.95)
        if idx == 0:
            row[0].set_title('Real', fontsize=14)
            row[1].set_title('Fake', fontsize=14)
            row[2].set_title('abs(Real - Fake)', fontsize=14)
        row[0].set_ylabel(param_str(params[idx]), labelpad=10, fontsize=12)
        idx = idx + 1
        scores.append([np.linalg.norm(c_r - c_f)])
    return np.array(scores)


# Compute the PSD correlation matrices for a list of real and fake images
# real_images and fake_images are two dimensional lists of samples in the form [[images for params p1], [images for params p2],  ...]
# eventually a function that loads the images can also be provided instead, e.g. [loader_params_p1, loader_params_p2, ...] to save memory
# params is a list representing the parameters [p1, p2, ...]
# box_l is the resolution of the images in sky degrees (radians)
# bin_k is the number of bins
# log_sampling is a flag that indicates whether the bins should be logarithmically samples or not
# cut is either None or in the format [a, b], where all the frequencies below a and above b are cut
# param_str is a function that takes the list of the current parameters and returns a label
# an array containing the l2 pairwise differences between real and fake images is returned
def compute_plot_correlations(real_images, fake_images, params, box_l=(5*np.pi/180), bin_k=50, cut=None, log_sampling=False, param_str=(lambda x: str(x)), tick_every=10, figsize=None, lenstools=False):

    # Compute correlations
    corr, k = compute_correlations(real_images, fake_images, params, box_l, bin_k, cut, log_sampling, lenstools=lenstools)

    # Plot and return
    return plot_correlations(corr, k, params, param_str, tick_every, figsize)


def interpolate_between(path, frames, has_stats=False):
    new_path = []
    for i in range(len(path) - 1):
        linspaces = []
        for j in range(len(path[i]) - 1):
            linspaces.append(np.linspace(path[i][j], path[i + 1][j], frames))
        for j in range(frames):
            new_p = []
            for k in range(len(linspaces)):
                new_p.append(linspaces[k][j])
            if j == 0 and i >= 1:
                continue # Path was already added before
            if j == 0:
                new_p.append(path[i][len(linspaces)])
            elif j == frames - 1:
                new_p.append(path[i + 1][len(linspaces)])
            else:
                new_p.append(has_stats)
            new_path.append(new_p)
    return new_path


# Real and fake are lists of arrays corresponding to sets of samples produced by different parameters
def compute_ssim_score(fake, real=None, gaussian_weights=True, sigma=1.5, ncopm=100):

    # Local function to compute scores of either fake or real
    def compute_single(X):
        scores = []
        for i in range(len(X)):
            imgs = X[i]
            if callable(X[i]):
                imgs = X[i]()
            scores.append(stats.ms_ssim(imgs, gaussian_weights=gaussian_weights, sigma=sigma, ncopm=ncopm))
        return np.array(scores)

    # Return scores for real and fake
    return [compute_single(fake), None if real is None else compute_single(real)]


# Compute frechet inception distances given real and fake activations
# real and fake are lists containing activations for different parameters
def compute_fids_from_activations(real, fake):
    fids = []
    for i in range(len(real)):
        fids.append(stats.compute_fid_from_activations(real[i], fake[i]))
    return np.array(fids)

# TODO: impossible to load images dynamically after regressor is instantiated
def compute_plot_fid(real_images, fake_images, params, regressor_path, reg_class="Regressor", axes=['$\Omega_M$', '$\sigma_8$'], lims=[None, None], batch_size=None, checkpoint=None, alpha=0.05):

    # Load regressor
    reg = load_regressor(regressor_path, reg_class=reg_class)

    # Function to load images into memory if necessary
    def load_imgs(X):
        if callable(X):
            return X()
        return X

    # Get regressor outputs for both real and fake images
    features = []
    pred_params = []
    for i in range(len(params)):
        imgs = load_imgs(real_images[i])
        curr_params = np.tile(params[i], [len(imgs), 1])
        ro = get_regressor_outputs(reg, imgs, curr_params, batch_size=batch_size, checkpoint=checkpoint)
        imgs = load_imgs(fake_images[i])
        fo = get_regressor_outputs(reg, imgs, curr_params, batch_size=batch_size, checkpoint=checkpoint)
        pred_params.append([ro[0], fo[0]])
        features.append([ro[1], fo[1]])
    features = np.array(features)
    pred_params = np.array(pred_params)

    # Compute frechet inception distances
    fids = compute_fids_from_activations(features[:, 0], features[:, 1])

    # Plot predicted params
    if len(params) == 1:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
        get_subplot = lambda x: ax
    elif np.sqrt(len(params)).is_integer():
        inter = int(np.sqrt(len(params)))
        fig, ax = plt.subplots(nrows=inter, ncols=inter, figsize=(15, 15))
        get_subplot = lambda x: ax[x//inter][x%inter]
    elif len(params) % 2 == 0:
        inter = int(np.sqrt(len(params)))
        fig, ax = plt.subplots(nrows=len(params)//2, ncols=2, figsize=(10, 5 * (len(params)//2)))
        get_subplot = lambda x: ax[x//2][x%2]
    else:
        fig, ax = plt.subplots(nrows=len(params), ncols=1, figsize=(5, 5 * len(params)))
        get_subplot = lambda x: ax[x]
    for i in range(len(params)):
        a = get_subplot(i)
        a.scatter(pred_params[i, 0, :, 0], pred_params[i, 0, :, 1], label="Real", alpha=alpha)
        a.scatter(pred_params[i, 1, :, 0], pred_params[i, 1, :, 1], label="Fake", color="r", alpha=alpha)
        a.set_title(axes[0] + ': ' + str(params[i][0]) + ', ' + axes[1] + ': ' + str(params[i][1]), fontsize=14)
        a.set_xlabel(axes[0], fontsize=12)
        a.set_ylabel(axes[1], fontsize=12)
        if lims[0] is not None:
            a.set_xlim(lims[0])
        if lims[1] is not None:
            a.set_ylim(lims[1])
        leg = a.legend(fontsize=12)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
    fig.tight_layout()

    return fids, fig




def plot_images_psd(images, title, filename=None, sigma_smooth=None):
    my_dpi = 200

    clip_max = 1e10

    images = np.clip(images, -1, clip_max)
    images = utils.makeit_square(images)

    n_rows = len(sigma_smooth)
    # n = n_rows*n_cols
    n = n_rows
    n_cols = 2
    # n_obs = images.shape[0]
    size_image = images.shape[1]
    m = max(5, size_image / my_dpi)
    plt.figure(figsize=(n_cols * m, n_rows * m), dpi=my_dpi)

    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.1, hspace=0.1)

    j = 0
    for i in range(n):
        # fig.add_subplot(gs[i]).set_xlabel(i)
        images1 = ndimage.gaussian_filter(images, sigma=sigma_smooth[i])
        ps_real, k = metric.power_spectrum_batch_phys(X1=images1)

        # PLOTING THE IMAGE
        ax = plt.subplot(gs[j])
        plot_array_plt(
            ndimage.gaussian_filter(images[1], sigma=sigma_smooth[i]),
            ax=ax,
            color='white')
        ax.set_ylabel(
            '$\sigma_{{smooth}} = {}$'.format(sigma_smooth[i]), fontsize=10)
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }

        # PSD
        ax1 = plt.subplot(gs[j + 1])
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        plot_with_shade(
            ax1,
            k,
            ps_real,
            color='b',
            label="Real $\mathcal{F}(X))^2$",
            **linestyle)
        ax1.set_ylim(bottom=0.1)
        if i == 0:
            ax1.title.set_text("2D Power Spectrum\n")
            ax1.title.set_fontsize(11)

        ax1.tick_params(axis='both', which='major', labelsize=10)
        if i == n - 1:
            ax1.set_xlabel("$k$", fontsize=10)
        else:
            ax1.set_xticklabels(())
        j += 2
        # ax1.set_aspect('equal')

    if filename is not None:

        filename = os.path.join('', '{}.png'.format(filename))
        plt.savefig(
            filename, bbox_inches='tight', dpi=my_dpi
        )  # bbox_extra_artists=(txt_top)) #, txt_left))  # Save Image
    plt.show()


# Plot a single curve
def plot_single(x, y, color='b', ax=None, label=None, xscale='linear', yscale='log', xlabel="", ylabel="", title="", shade=False, confidence=None, xlim=None, ylim=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    linestyle = {
        "linewidth": 1,
        "markeredgewidth": 0,
        "markersize": 3,
        "marker": "o",
        "linestyle": "-"
    }
    if shade:
        plot_with_shade(ax, x, y, label=label, color=color, confidence=confidence, **linestyle)
    else:
        if confidence is not None:
            y = np.mean(y, axis=0)
        ax.plot(x, y, label=label, color=color, **linestyle)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, list):
            ylim = ylim[0]
        ax.set_ylim(ylim)
    ax.title.set_text(title + "\n")
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if label:
        ax.legend(fontsize=14, loc=3)


# Comparison plot between two curves real and fake corresponding to axis x
def plot_cmp(x, fake, real=None, xscale='linear', yscale='log', xlabel="", ylabel="", ax=None, title="", shade=False, confidence=None, xlim=None, ylim=None, fractional_difference=False, algorithm='classic', loc=3):
    if ax is None:
        ax = plt.gca()
    if real is None:
        # Plot a line that will be canceled by fake
        # In this way the fractional difference is zero
        plot_single(x, fake, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=False, confidence=confidence, xlim=xlim, ylim=ylim)
    else:
        plot_single(x, real, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    plot_single(x, fake, color='r', ax=ax, label="Fake", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=14, loc=loc)
    if fractional_difference:
        if real is None:
            real = fake
        if shade:
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        plot_fractional_difference(x, real, fake, xscale=xscale, yscale='linear', ax=ax, color='g', ylim=ylim[1] if isinstance(ylim, list) else None, algorithm=algorithm, loc=loc)
        
        
# Plot an image
def plot_img(img, x=None, title="", ax=None, cmap=plt.cm.plasma, vmin=None, vmax=None, tick_every=10, colorbar=False, log_norm=False):
    if ax is None:
        ax = plt.gca()
    img = np.squeeze(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=LogNorm() if log_norm else None)
    ax.set_title(title)
    ax.title.set_fontsize(16)
    if x is not None:
        # Define axes
        ticklabels = []
        for i in range(len(x)):
            if i % tick_every == 0:
                ticklabels.append(str(int(round(x[i], 0))))
        ticks = np.linspace(0, len(x) - (len(x) % tick_every), len(ticklabels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
    else:
        ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im


# Plot fractional difference
# if algorithm='classic' then the classic fractional difference abs(real - fake) / real is computed
# if algorithm='relative' then abs(real-fake) / std(real) is computed
def plot_fractional_difference(x, real, fake, xlim=None, ylim=None, xscale="log", yscale="linear", title="", ax=None, color='b', algorithm='classic', loc=3):
    if ax is None:
        ax1 = plt.gca()
    else:
        ax1 = ax.twinx()
    diff = np.abs(real - fake)
    if algorithm == 'classic':
        diff = diff / real
    elif algorithm == 'relative':
        diff = diff / np.std(real, axis=0)
    else:
        raise ValueError("Unknown algorithm name " + str(algorithm))
    plot_single(x, diff, ax=ax1, xscale=xscale, yscale=yscale, title=title, color=color, label='Frac. diff.' if algorithm == 'classic' else 'Rel. diff.', ylim=ylim)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color)

    # Adjust legends
    if ax is not None:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax.get_legend().remove()
        ax1.legend(lines + lines2, labels + labels2, loc=loc, fontsize=14)


def plot_histogram(x, histo, yscale='log', tick_every=10, bar_width=1):
    plt.bar(np.arange(len(histo)), histo, bar_width)
    positions, labels = ([], [])
    for idx in range(len(x)):
        if idx == len(x) - 1:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
        if idx % tick_every == 0:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
    plt.xticks(positions, labels)
    plt.yscale(yscale)


# Plot the scores as a heatmap on the parameter grid
# test_scores: array containing the scores for different parameters
# test_params: array of shape (n, 2) containing the parameters
# thresholds: if None a single heatmap with continuous values is produced
#             if a list is provided a subplot containing different discrete heatmaps 
#             (red points if below threshold and green otherwise) are produced
def plot_heatmap(test_scores, test_params, train_scores=None, train_params=None, vmin=0, vmax=0.25, thresholds=None, labels=['$\Omega_M$', '$\sigma_8$']):
    if thresholds is None:
        plt.scatter(test_params[:, 0], test_params[:, 1], c=test_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k')
        if train_scores is not None and train_params is not None:
            plt.scatter(train_params[:, 0], train_params[:, 1], c=train_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k', marker='s')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.colorbar()
    else:
        _, ax = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(len(thresholds) * 5, 5))
        for j in range(len(thresholds)):
            if train_params is not None and train_scores is not None:
                for i in range(len(train_params)):
                    ax[j].scatter(train_params[i, 0], train_params[i, 1], c='g' if train_scores[i] <= thresholds[j] else 'r', marker='s')
            for i in range(len(test_params)):
                ax[j].scatter(test_params[i, 0], test_params[i, 1], c='g' if test_scores[i] <= thresholds[j] else 'r')
            ax[j].set_xlabel(labels[0])
            ax[j].set_ylabel(labels[1])
            ax[j].set_title('Threshold: ' + str(thresholds[j]))