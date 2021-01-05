#!/usr/bin/env python

# clustering algorithms
from sklearn.cluster import dbscan

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

from array_plotting import plotting

import numpy as np

# import python packages
import obspy
import scipy
import seaborn as sns
from matplotlib.patches import Ellipse  # for drawing error ellipse

# import things for calculating means and standard deviations of angles.
from scipy.stats import circmean, circstd, circvar, gaussian_kde
from scipy.spatial import distance

import os
import shutil

from Parameters_Bootstrap import *

import circ_array as ca
from cluster_utilities import cluster_utilities


st = obspy.read(filepath)

distances = ca.get_distances(st, type="deg")

# get predicted slownesses and backazimuths
predictions = ca.pred_baz_slow(stream=st, phases=phases, one_eighty=True)


# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[
    row, :
][0]
PRED_BAZ_X = float(PRED_BAZ_X)
PRED_BAZ_Y = float(PRED_BAZ_Y)

All_Noise_arr = np.load(
    numpy_dir + "All_Noise_%s_%s_%s_array.npy" % (Boots, fmin, fmax)
)
All_Thresh_Peaks_arr = np.load(
    numpy_dir + "All_Thresh_Peaks_%s_%s_%s_array.npy" % (Boots, fmin, fmax)
)
Lin_Mean = np.load(numpy_dir + "Mean_Lin_%s_%s_%s_array.npy" % (Boots, fmin, fmax))

core_samples, labels = dbscan(
    X=All_Thresh_Peaks_arr, eps=epsilon, min_samples=int(Boots * MinPts)
)
no_clusters = np.amax(labels) + 1
cu = cluster_utilities(labels=labels, points=All_Thresh_Peaks_arr)

means_xy, means_baz_slow = cu.cluster_means()
bazs_std, slows_std = cu.cluster_std_devs()
ellipse_areas = cu.cluster_ellipse_areas(std_dev=2)

x_min = PRED_BAZ_X + slow_min
x_max = PRED_BAZ_X + slow_max
y_min = PRED_BAZ_Y + slow_min
y_max = PRED_BAZ_Y + slow_max


# write to results file

Results_filepath = Res_dir + f"Clustering_Results_{fmin:.2f}_{fmax:.2f}.txt"

# make new lines list

newlines = cu.create_newlines(
    st,
    file_path=Results_filepath,
    phase=phase,
    window=[t_min, t_max],
    Boots=Boots,
    epsilon=epsilon,
    slow_vec_error=3,
    Filter=False,
)

# plot!

with PdfPages(Res_dir + f"Clustering_Summary_Plot_{fmin:.2f}_{fmax:.2f}.pdf") as pdf:
    # clusters
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    p = plotting(ax=ax1)

    p.plot_clusters_XY(
        labels=labels,
        tp=Lin_Mean,
        peaks=All_Thresh_Peaks_arr,
        sxmin=x_min,
        sxmax=x_max,
        symin=y_min,
        symax=y_max,
        sstep=s_space,
        title="Clusters",
        predictions=predictions,
        ellipse=True,
        std_devs=[1, 2, 3],
    )

    pdf.savefig()
    plt.close()

    # record section
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    p = plotting(ax=ax2)
    p.plot_record_section_SAC(
        st=st, phase=phase, tmin=cut_min, tmax=cut_max, align=True
    )

    ax2.vlines(
        x=t_min, ymin=np.amin(distances), ymax=np.amax(distances), label="window min"
    )

    ax2.vlines(
        x=t_max, ymin=np.amin(distances), ymax=np.amax(distances), label="window min"
    )

    pdf.savefig()
    plt.close()

    ### plot stations

    fig = plt.figure(figsize=(6, 6))

    # need to give a projection for the cartopy package
    # to work. See a list here:
    # https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    p = plotting(ax=ax)
    p.plot_stations(st)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    pdf.savefig()
    plt.close()

    ### plot great circle path

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())

    p = plotting(ax=ax)
    p.plot_paths(st)

    pdf.savefig()
    plt.close()
