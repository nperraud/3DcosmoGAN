def plot_array_pil(x,
                   pil_object=True,
                   lims=None,
                   log_input=False,
                   log_clip=1e-7):

    norm_x = np.squeeze(x)

    cm = planck_cmap()
    plt.register_cmap(cmap=cm)

    img = Image.fromarray(cm(norm_x, bytes=True))

    if pil_object:
        return img
    else:
        return np.array(img)

def plot_array_plt(x, ax=None, cmap='planck', color='black', simple_k=10):
    if cmap == 'planck':
        cmap = planck_cmap()
        plt.register_cmap(cmap=cmap)

    x = x.reshape(-1)
    size = int(len(x)**0.5)

    log_x = utils.forward_map(x, simple_k)
    lims = [-1, 1]
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.axis([0, size, 0, size])
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')  # labels along the bottom edge are off
    try:
        [i.set_color(color) for i in ax.spines.itervalues()]
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
    except:
        [i.set_color(color) for i in ax.spines.values()]
        [i.set_linewidth(2) for i in ax.spines.values()]

    im = plt.pcolormesh(
        np.reshape(log_x, [size, size]),
        cmap=cmap,
        vmin=lims[0],
        vmax=lims[1],
        edgecolors='face')

    return im


def planck_cmap(ncolors=256):
    """
    Returns a color map similar to the one used for the "Planck CMB Map".
    Parameters
    ----------
    ncolors : int, *optional*
    Number of color segments (default: 256).
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap instance
    Linear segmented color map.
    """
    segmentdata = {
        "red": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                (0.3, 0.00, 0.00), (0.4, 0.00, 0.00), (0.5, 1.00, 1.00),
                (0.6, 1.00, 1.00), (0.7, 1.00, 1.00), (0.8, 0.83, 0.83),
                (0.9, 0.67, 0.67), (1.0, 0.50, 0.50)],
        "green": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                  (0.3, 0.30, 0.30), (0.4, 0.70, 0.70), (0.5, 1.00, 1.00),
                  (0.6, 0.70, 0.70), (0.7, 0.30, 0.30), (0.8, 0.00, 0.00),
                  (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)],
        "blue": [(0.0, 0.50, 0.50), (0.1, 0.67, 0.67), (0.2, 0.83, 0.83),
                 (0.3, 1.00, 1.00), (0.4, 1.00, 1.00), (0.5, 1.00, 1.00),
                 (0.6, 0.00, 0.00), (0.7, 0.00, 0.00), (0.8, 0.00, 0.00),
                 (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)]
    }
    return cm("Planck-like", segmentdata, N=int(ncolors), gamma=1.0)


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