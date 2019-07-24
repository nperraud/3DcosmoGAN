"""This module contains the different statistic functions."""

import numpy as np
from . import power_spectrum_phys as ps
import scipy.ndimage.filters as filters
from skimage.measure import compare_ssim
import itertools
from gantools import utils
import functools
import multiprocessing as mp
import tensorflow as tf



def wrapper_func(x, bin_k=50, box_l=100 / 0.7,
                 log_sampling=True):
    tmp = ps.dens2overdens(np.squeeze(x), np.mean(x))
    return ps.power_spectrum(field_x=tmp, box_l=box_l, bin_k=bin_k, log_sampling=log_sampling)[0]


def wrapper_func_cross(a,
                       X2,
                       self_comp,
                       sx,
                       sy,
                       sz=None,
                       bin_k=50,
                       box_l=100 / 0.7,
                       is_3d=False):
    inx, x = a
    _result = []
    for iny, y in enumerate(X2):
        # if it is a comparison with it self only do the low triangular matrix
        if (self_comp and (inx < iny)) or not self_comp:
            if is_3d:
                over_dens_x = ps.dens2overdens(x.reshape(sx, sy, sz))
                over_dens_y = ps.dens2overdens(y.reshape(sx, sy, sz))
            else:
                over_dens_x = ps.dens2overdens(x.reshape(sx, sy))
                over_dens_y = ps.dens2overdens(y.reshape(sx, sy))
            tmp = ps.power_spectrum(
                field_x=over_dens_x,
                box_l=box_l,
                bin_k=bin_k,
                field_y=over_dens_y)[0]
            # Nati: Why is there a [0] here. There is probably a good reason...
            _result.append(tmp)
    return _result


def power_spectrum_batch_phys(X1,
                              X2=None,
                              bin_k=50,
                              box_ll=350, #100 / 0.7,
                              resolution=256,
                              remove_nan=True,
                              is_3d=False,
                              multiply=False,
                              log_sampling=True,
                              cut=None):
    """
    Calculates the 1-D PSD of a batch of variable size
    :param batch:
    :param size_image:
    :param multiply: whether the psd should be multiply by k(k+1)/2pi
    :return: result, k
    """

    if len(X1.shape)==5 or (len(X1.shape)==4 and not(X1.shape[3]==1)):
        is_3d = True
    else:
        is_3d = False
    sx, sy = X1[0].shape[0], X1[0].shape[1]
    sz = None
    if is_3d:
        sz = X1[0].shape[2]
    if not (sx == sy):
        X1 = utils.makeit_square(X1)
        s = X1[0].shape[0]
    else:
        s = sx
        # ValueError('The image need to be squared')
    
    # Compute the good value of box_l
    box_l = box_ll*sx/resolution
    
    if is_3d:
        _, k = ps.power_spectrum(
            field_x=X1[0].reshape(s, s, s), box_l=box_l, bin_k=bin_k, log_sampling=log_sampling)
    else:
        _, k = ps.power_spectrum(
            field_x=X1[0].reshape(s, s), box_l=box_l, bin_k=bin_k, log_sampling=log_sampling)

    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        if X2 is None:
            # # Pythonic version
            # over_dens = [ps.dens2overdens(x.reshape(s, s), np.mean(x)) for x in X1]
            # result = np.array([
            #     ps.power_spectrum(field_x=x, box_l=box_l, bin_k=bin_k)[0]
            #     for x in over_dens
            # ])
            # del over_dens

            # Make it multicore...
            func = functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k, log_sampling=log_sampling)
            result = np.array(pool.map(func, X1))

        else:
            if not (sx == sy):
                X2 = utils.makeit_square(X2)
            self_comp = np.all(X2 == X1)
            _result = []
            # for inx, x in enumerate(X1):
            #     # for iny, y in enumerate(X2):
            #     #     # if it is a comparison with it self only do the low
            #     #     # triangular matrix
            #     #     if (self_comp and (inx < iny)) or not self_comp:
            #     #         over_dens_x = ps.dens2overdens(x.reshape(sx, sy))
            #     #         over_dens_y = ps.dens2overdens(y.reshape(sx, sy))
            #     _result += wrapper_func_cross(
            #         (inx, x), X2, self_comp, sx, sy, bin_k=50, box_l=100/0.7)
            func = functools.partial(
                wrapper_func_cross,
                X2=X2,
                self_comp=self_comp,
                sx=sx,
                sy=sy,
                sz=sz,
                bin_k=bin_k,
                box_l=box_l,
                is_3d=is_3d,
                log_sampling=log_sampling)
            _result = pool.map(func, enumerate(X1))
            _result = list(itertools.chain.from_iterable(_result))
            result = np.array(_result)

    if remove_nan:
        # Some frequencies are not defined, remove them
        freq_index = ~np.isnan(result).any(axis=0)
        result = result[:, freq_index]
        k = k[freq_index]

    if cut is not None:
        # Cut lower frequencies out
        idx = 0
        while idx < len(k) - 1 and k[idx] < cut[0]:
            idx = idx + 1
        idx_low = idx
        # Cut higher frequencies out
        while idx < len(k) - 1 and k[idx] < cut[1]:
            idx = idx + 1
        k = k[idx_low:idx]
        result = result[:, idx_low:idx]

    if multiply:
        result = result * (k + 1) * k / (2 * np.pi) 

    return result, k


def psd_correlation(x, bin_k=50, box_l=100 / 0.7, log_sampling=True, cut=None):
    psd, k = power_spectrum_batch_phys(x, bin_k=bin_k, box_l=box_l, log_sampling=log_sampling, cut=cut)
    return np.corrcoef(psd, rowvar=False), k


def histogram(x, bins, probability=True):
    if x.ndim > 2:
        x = np.reshape(x, [int(x.shape[0]), -1])

    edges = np.histogram(x[0].ravel(), bins=bins)[1][:-1]

    counts = np.array([np.histogram(y, bins=bins)[0] for y in x])

    if probability:
        density = counts * 1.0 / np.sum(counts, axis=1, keepdims=True)
    else:
        density = counts

    return edges, density


def peak_count(X, neighborhood_size=5, threshold=0.5):
    """
    Peak cound for a 2D or a 3D square image
    :param X: numpy array shape [n,n] or [n,n,n]
    :param neighborhood_size: size of the local neighborhood that should be filtered
    :param threshold: minimum distance betweent the minimum and the maximum to be considered a local maximum
                      Helps remove noise peaks (0.5 since the number of particle is supposed to be an integer)
    :return: vector of peaks found in the array (int)
    """
    size = len(X.shape)
    if len(X.shape) == 1:
        pass
    elif size==2:
        assert(X.shape[0]==X.shape[1])
    elif size==3:
        assert(X.shape[0]==X.shape[1]==X.shape[2])
    else:
        raise Exception(" [!] Too many dimensions")

    # PEAK COUNTS
    data_max = filters.maximum_filter(X, neighborhood_size)
    maxima = (X == data_max)
    data_min = filters.minimum_filter(X, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    return np.extract(maxima, X)



def chi2_distance(peaksA, peaksB, eps=1e-10, **kwargs):
    histA, _ = np.histogram(peaksA, **kwargs)
    histB, _ = np.histogram(peaksB, **kwargs)

    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b)**2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


def distance_chi2_peaks(im1, im2, bins=100, range=[0, 2e5], **kwargs):
    if len(im1.shape) > 2:
        im1 = im1.reshape(-1)
    distance = []

    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        for x in im1:
            # for y in im2:
            #     distance.append(chi2_distance(x, y, bins=bins, range=range, **kwargs))
            distance.append(
                np.array(
                    pool.map(
                        functools.partial(
                            chi2_distance,
                            peaksB=x,
                            bins=bins,
                            range=range,
                            **kwargs), im2)))
    return np.mean(np.array(distance))



def psd_metric(gen_sample_raw, real_sample_raw):
    psd_gen, _ = power_spectrum_batch_phys(X1=gen_sample_raw)
    psd_real, _ = power_spectrum_batch_phys(X1=real_sample_raw)
    psd_gen = np.mean(psd_gen, axis=0)
    psd_real = np.mean(psd_real, axis=0)
    return diff_vec(psd_real, psd_gen)


def diff_vec(y_real, y_fake):
    e = y_real - y_fake
    l2 = np.mean(e * e)
    l1 = np.mean(np.abs(e))
    loge = 10 * (np.log10(y_real + 1e-2) - np.log10(y_fake + 1e-2))
    logel2 = np.mean(loge * loge)
    logel1 = np.mean(np.abs(loge))
    return l2, logel2, l1, logel1


# Returns the fractional difference
def fractional_diff(y_real, y_fake, axis=0):
    return np.abs(y_real - y_fake) / y_real

# Returns the difference divided by the standard deviation
def relative_diff(y_real, y_fake, axis=0):
    return np.abs(np.mean(y_real, axis=axis) - np.mean(y_fake, axis=axis)) / np.std(y_real, axis=axis)


def peak_count_hist(dat, bins=20, lim=None, neighborhood_size=5, threshold=0, log=True, mean=True):
    """Make the histogram of the peak count of data.

    Arguments
    ---------
    dat  : input data (numpy array, first dimension for the sample)
    bins : number of bins for the histogram (default 20)
    lim  : limit for the histogram, if None, then min(peak), max(peak)
    """
    
    # Remove single dimension...
    dat = np.squeeze(dat)
    
    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        peak_count_arg = functools.partial(peak_count, neighborhood_size=neighborhood_size, threshold=threshold)
        peak_arr = np.array(pool.map(peak_count_arg, dat))
    peak = np.hstack(peak_arr)
    if log:
        peak = np.log(peak+np.e)
        peak_arr = np.array([np.log(pa+np.e) for pa in peak_arr])
    if lim is None:
        lim = (np.min(peak), np.max(peak))
    else:
        lim = tuple(map(type(peak[0]), lim))
    # Compute histograms individually
    with mp.Pool(processes=num_workers) as pool:
        hist_func = functools.partial(unbounded_histogram, bins=bins, range=lim)
        res = np.array(pool.map(hist_func, peak_arr))
    
    # Unpack results
    y = np.vstack(res[:, 0])
    x = res[0, 1]

    x = (x[1:] + x[:-1]) / 2
    if log:
        x = np.exp(x)-np.e
    if mean:
        y = np.mean(y, axis=0)
    return y, x, lim


def peak_count_hist_real_fake(real, fake, bins=20, lim=None, log=True, neighborhood_size=5, threshold=0, mean=True):
    y_real, x, lim = peak_count_hist(real, bins=bins, lim=lim, log=log, neighborhood_size=neighborhood_size, threshold=threshold, mean=mean)
    y_fake, _, _ = peak_count_hist(fake, bins=bins, lim=lim, log=log, neighborhood_size=neighborhood_size, threshold=threshold, mean=mean)
    return y_real, y_fake, x

def unbounded_histogram(dat, range=None, **kwargs):
    if range is None:
        return np.histogram(dat, **kwargs)
    y, x = np.histogram(dat, range=range, **kwargs)
    y[0] = y[0] + np.sum(dat<range[0])
    y[-1] = y[-1] + np.sum(dat>range[1])
    return y, x

def mass_hist(dat, bins=20, lim=None, log=True, mean=True, **kwargs):
    """Make the histogram of log10(data) data.

    Arguments
    ---------
    dat  : input data
    bins : number of bins for the histogram (default 20)
    lim  : limit for the histogram, if None then min(log10(dat)), max(dat)
    """
    if log:
        log_data = np.log10(dat + 1)
    else:
        log_data = dat
    if lim is None:
        lim = (np.min(log_data), np.max(log_data))

    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        results = [pool.apply(unbounded_histogram, (x,), dict(bins=bins, range=lim)) for x in log_data]
    y = np.vstack([y[0] for y in results])
    x = results[0][1]
    if log:
        x = 10**((x[1:] + x[:-1]) / 2) - 1
    else:
        x = (x[1:] + x[:-1]) / 2
    if mean:
        return np.mean(y, axis=0), x, lim
    else:
        return y, x, lim


def mass_hist_real_fake(real, fake, bins=20, lim=None, log=True, mean=True):
    if lim is None:
        new_lim = True
    else:
        new_lim = False
    y_real, x, lim = mass_hist(real, bins=bins, lim=lim, log=log, mean=mean)
    if new_lim:
        lim = list(lim)
        lim[1] = lim[1]+1
        y_real, x, lim = mass_hist(real, bins=bins, lim=lim, log=log, mean=mean)

    y_fake, _, _ = mass_hist(fake, bins=bins, lim=lim, log=log, mean=mean)
    return y_real, y_fake, x



def total_stats_error(feed_dict, params=dict()):
    """Generate a weighted total loss based on the image PSD, Mass and Peak
    histograms"""
    if isinstance(params, list):
        if len(params) == 2:
            params = dict(
                w_l1_log_psd=params[0],
                w_l2_log_psd=params[1],
                w_l1_log_mass_hist=params[0],
                w_l2_log_mass_hist=params[1],
                w_l1_log_peak_hist=params[0],
                w_l2_log_peak_hist=params[1]
            )
        elif len(params) == 7:
            params = dict(
                w_l1_log_psd = params[0],
                w_l2_log_psd = params[1],
                w_l1_log_mass_hist = params[2],
                w_l2_log_mass_hist = params[3],
                w_l1_log_peak_hist = params[4],
                w_l2_log_peak_hist = params[5],
                w_wasserstein_mass_hist = params[6]
            )
        else:
            raise Exception(" [!] If total_stat_error params are specified as a list,"
                            " length must be either 2 or 7")

    v = 0
    v += params.get("w_l1_log_psd", 0) * feed_dict['log_l1_psd']
    v += params.get("w_l2_log_psd", 1) * feed_dict['log_l2_psd']
    v += params.get("w_l1_log_mass_hist", 0) * feed_dict['log_l1_mass_hist']
    v += params.get("w_l2_log_mass_hist", 1) * feed_dict['log_l2_mass_hist']
    v += params.get("w_l1_log_peak_hist", 0) * feed_dict['log_l1_peak_hist']
    v += params.get("w_l2_log_peak_hist", 1) * feed_dict['log_l2_peak_hist']
    v += params.get("w_wasserstein_mass_hist", 0)\
         * np.log10(feed_dict['wasserstein_mass_hist'] + 1)

    return v

# Note: this only works for 2D images
# X is an array of shape (nsamples, x, y)
def ms_ssim(X, gaussian_weights=True, sigma=1.5, ncopm=100):
    assert(len(X) >= 2)
    if len(X.shape) > 3:
        X = X[:, :, :, 0]
    msssim = 0
    for i in range(ncopm):
        a, b = np.random.randint(0, len(X)), np.random.randint(0, len(X))
        while a == b:
            b = np.random.randint(0, len(X))
        msssim = msssim + compare_ssim(X[a], X[b])
    return msssim / ncopm

# Compute frechet inception distance from activations
def compute_fid_from_activations(f1, f2):
    with tf.Session() as sess:
        return sess.run([tf.contrib.gan.eval.frechet_classifier_distance_from_activations(tf.constant(f1), tf.constant(f2))])[0]


# TODO: remove once correlation code is fixed
import lenstools
import astropy

def psd_single_lenstools(img, angle=(5*np.pi/180), l_edges=[200, 6000, 50], multiply=False):
    if len(img.shape) > 2:
        img = img[:, :, 0]
    c = lenstools.ConvergenceMap(data=img, angle=angle * astropy.units.rad)
    l, psd = c.powerSpectrum(l_edges)
    if multiply:
        psd = psd * l * (l+1) / (2 * np.pi)
    return psd, l

def psd_lenstools(data, box_l=(5*np.pi/180), bin_k=50, cut=None, multiply=False):
    psd = []
    if cut is None:
        cut = [200, 6000]
    l_edges = np.linspace(cut[0], cut[1], bin_k)
    for img in data:
        p, l = psd_single_lenstools(img, angle=box_l, l_edges=l_edges, multiply=multiply)
        psd.append(p)
    return np.array(psd), l

def psd_correlation_lenstools(x, box_l=(5*np.pi/180), bin_k=50, cut=None):
    psd, k = psd_lenstools(x, box_l=box_l, bin_k=bin_k, cut=cut)
    return np.corrcoef(psd, rowvar=False), k