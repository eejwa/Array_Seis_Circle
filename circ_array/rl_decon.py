import numpy as np 


# this has been taken from the sci-kit image package and edited slightly
def RL_decon(image, psf, iterations=10, clip=True):
    """Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
    Input degraded image (can be N dimensional).
    psf : ndarray
    The point spread function.
    iterations : int
    Number of iterations. This parameter plays the role of
    regularisation.
    clip : boolean, optional
    True by default. If true, pixel value of the result above 1 or
    under -1 are thresholded for skimage pipeline compatibility.
    Returns
    -------
    im_deconv : ndarray
    The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
        print("Frequency domain")
    else:
        convolve_method = convolve
        print("Time domain")

# Here is the deconvolution
# set image and psf as arrays of floats
    image = image.astype(np.float)
    psf = psf.astype(np.float)

    # im_deconv = image
    im_deconv = image

    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'valid')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'valid')
    return im_deconv
