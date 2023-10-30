#!/usr/bin/env python 
import random 
import numpy as np 
import scipy 
from beamforming_xy import BF_Noise_Threshold_Relative_XY
from extract_peaks import findpeaks_XY
import numba as nb

@nb.jit(nopython=True)
def bootstrap_beamform_xy(traces, slow_min_x, slow_max_x, slow_min_y,
                          slow_max_y, slow_space, sampling_rate, geometry, distances, nboots):
    """
    
    """

    # tps = []
    # noises = []
    # peakss = []
    nsx = int(np.round(((slow_max_x - slow_min_x) / slow_space), 0) + 1)
    nsy = int(np.round(((slow_max_y - slow_min_y) / slow_space), 0) + 1)

    # make empty array for output.
    tps = np.zeros((nboots, nsy, nsx))
    noises = np.zeros(nboots)


    for b in range(nboots):
        print(b)
        indices = np.arange(traces.shape[0])
        rand_indices = np.random.choice(indices,size=traces.shape[0])

        Traces_new = traces[rand_indices]
        geometry_new = geometry[rand_indices]
        distances_new = distances[rand_indices]
        avg_dist = np.mean(distances_new)

        # get centre
        centre_x, centre_y, centre_z = (
            np.mean(geometry_new[:, 0]),
            np.mean(geometry_new[:, 1]),
            np.mean(geometry_new[:, 2]),
        )

        tp, noise_mean, max_peak = BF_Noise_Threshold_Relative_XY(
            traces=Traces_new,
            sampling_rate=sampling_rate,
            geometry=geometry_new,
            distance=avg_dist,
            sxmin=slow_min_x,
            sxmax=slow_max_x,
            symin=slow_min_y,
            symax=slow_max_y,
            s_space=slow_space,
            elevation=False
        )

        # Smoothed_thresh_lin_array = scipy.ndimage.filters.gaussian_filter(
        #     tp, 1, mode="constant"
        # )

        # Threshold_lin_array = np.copy(tp)
        # Threshold_lin_array[tp <= noise_mean * 3] = 0


        # peaks, powers = findpeaks_XY(
        #     Threshold_lin_array,
        #     xmin=slow_min_x,
        #     xmax=slow_max_x,
        #     ymin=slow_min_y,
        #     ymax=slow_max_y,
        #     xstep=slow_space,
        #     ystep=slow_space,
        #     N=n_peaks
        # )
        tps[b] = tp
        noises[b] = noise_mean

    return
        # tps.append(tp)
        # noises.append(noise_mean)
        # peakss.append(peaks)

    # tps = np.array(tps)
    # noises = np.array(noises)
    # peakss = np.array(peakss)

    # return tps, noises, peakss
