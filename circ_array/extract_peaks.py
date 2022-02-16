
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def findpeaks_XY(Array, xmin, xmax, ymin, ymax, xstep, ystep, N=10):
    """
    Peak finding algorith for a 2D array of values. The peaks will be searched for
    within a range of points from a predicted arrival. Edited from stack overflow
    answer: https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array.

    Parameters
    ----------
    Array : 2-D numpy array of floats.
        2-D array of floats representing power or some other parameter.

    xmin : float
        Minumum x point of the area to search for peaks.

    sl_xmax : float
        Maximum x point of the area to search for peaks.

    sl_ymin : float
        Minumum y point of the area to search for peaks.

    sl_ymax : float
        Maximum y point of the area to search for peaks.

    step : float
        Increments of points in x/y axis used in the array.

    N : int
        The top N peaks will be returned.

    Returns
    -------
    peaks : 2D array of floats
        The top N peaks of the array of the format [[x,y]],
        where x and y are the coordinate locations in slowness
        space.
    power_vals : 1D array of floats
        The power of the respective peak in the peaks arrays.
    """

    # define space
    steps_x = int(np.round((xmax - xmin) / xstep, decimals=0)) + 1
    steps_y = int(np.round((ymax - ymin) / ystep, decimals=0)) + 1
    xpoints = np.linspace(xmin, xmax, steps_x, endpoint=True)
    ypoints = np.linspace(ymin, ymax, steps_y, endpoint=True)

    neighborhood = generate_binary_structure(2, 3)

    # maximum filter will take the array 'Array'
    # For each point it will take the adjacent grid points and find the highest ones
    # In other words, it will find the local maxima.
    # local_max = maximum_filter(Array, 3)
    local_max = maximum_filter(Array, footprint=neighborhood, mode='nearest') == Array
    # local_min = minimum_filter(Array_New, 3)

    background = Array == 0

    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # OK I'll try and find the gradient at each point then take the lowest values.

    detected_peaks = local_max ^ eroded_background

    y_peak, x_peak = np.where(detected_peaks == 1)

    # get top N values
    val_points = np.array([Array[y_peak, x_peak], y_peak, x_peak]).T

    val_points_sorted = np.copy(val_points)

    # sort arguments based on the first column
    # Just like Python, in that [::-1] reverses the array returned by argsort()
    #  [:N] gives the first N elements of the reversed list
    val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort()][::-1][:N]

    # get x and y locations of top N points
    val_x_points_sorted = val_points_sorted[:, 2]
    val_y_points_sorted = val_points_sorted[:, 1]

    # get power values of top N points
    val_points_power = val_points_sorted[:, 0]

    # find this location in slowness space
    x_peaks_space = xmin + (x_peak * xstep)
    y_peaks_space = ymin + (y_peak * ystep)

    x_vals_peaks_space = xmin + (val_x_points_sorted * xstep)
    y_vals_peaks_space = ymin + (val_y_points_sorted * ystep)

    peaks_combined = np.array((x_peaks_space, y_peaks_space)).T
    peaks_combined_vals = np.array((x_vals_peaks_space, y_vals_peaks_space)).T

    return peaks_combined_vals, val_points_power


def findpeaks_Pol(Array, smin, smax, bmin, bmax, sstep, bstep, N=10):
    """
    Peak finding algorith for a 2D array of values. The peaks will be searched for
    within a range of points from a predicted arrival. This is edited for the polar
    coordinate search output.

    Parameters
    ----------
    Array : 2-D numpy array of floats
        2-D array of floats representing power or some other parameter.

    smin : float
        Minumum horizontal slowness.

    smax : float
        Maximum horizontal slowness.

    bmin : float
        Minumum backazimuth.

    bmax : float
        Maximum backazimuth.

    step : float
        Increments of slowness values.

    btep : float
        Increments of backazimuth values.

    N : int
        The top N peaks will be returned.

    Returns
    -------
    peaks : 2D array of floats
        The top N peaks of the array in the form of [baz,slow].
    """

    # define space
    steps_s = int(np.round((smax - smin) / sstep, decimals=0)) + 1
    steps_b = int(np.round((bmax - bmin) / bstep, decimals=0)) + 1

    spoints = np.linspace(smin, smax, steps_s, endpoint=True)
    bpoints = np.linspace(bmin, bmax, steps_b, endpoint=True)

    spoints = spoints
    bpoints = bpoints

    neighborhood = generate_binary_structure(2, 3)

    # maximum filter will take the array 'Array'
    # For each point it will take the adjacent grid points and find the highest ones
    # In other words, it will find the local maxima.
    local_max = maximum_filter(Array, footprint=neighborhood, mode='nearest') == Array

    background = Array == 0

    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # OK I'll try and find the gradient at each point then take the lowest values.

    detected_peaks = local_max ^ eroded_background

    s_peak, b_peak = np.where(detected_peaks == 1)

    # get top N values
    val_points = np.array([Array[s_peak, b_peak], s_peak, b_peak]).T

    val_points_sorted = np.copy(val_points)

    # sort arguments based on the first column
    # Just like Python, in that [::-1] reverses the array returned by argsort()
    #  [:N] gives the first N elements of the reversed list
    val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort()][::-1][:N]
    val_b_points_sorted = val_points_sorted[:, 2]
    val_s_points_sorted = val_points_sorted[:, 1]

    # this gives all the peaks in the array.
    b_peaks_space = bmin + (b_peak * bstep)
    s_peaks_space = smin + (s_peak * sstep)

    # get locations in the array
    b_vals_peaks_space = bmin + (val_b_points_sorted * bstep)
    s_vals_peaks_space = smin + (val_s_points_sorted * sstep)

    b_peaks_baz_vals = b_vals_peaks_space
    s_peaks_baz_vals = s_vals_peaks_space

    peaks_combined_vals = np.array((b_peaks_baz_vals, s_peaks_baz_vals)).T

    return peaks_combined_vals
