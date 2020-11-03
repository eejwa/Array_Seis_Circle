#!/usr/bin/env python

# clustering algorithms
from sklearn.cluster import dbscan

import matplotlib
import matplotlib.pyplot as plt

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

from circ_array import circ_array
ca = circ_array()

st = obspy.read(filepath)

# get array metadata
event_time = ca.get_eventtime(st)
geometry = ca.get_geometry(st)
distances = ca.get_distances(st,type='deg')
mean_dist = np.mean(distances)
stations = ca.get_stations(st)
centre_x, centre_y =  np.mean(geometry[:, 0]),  np.mean(geometry[:, 1])
sampling_rate=st[0].stats.sampling_rate
evdp = st[0].stats.sac.evdp
evlo = st[0].stats.sac.evlo
evla = st[0].stats.sac.evla

# get predicted slownesses and backazimuths
predictions = ca.pred_baz_slow(
    stream=st, phases=phases, one_eighty=True)

print(predictions)
# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]
PRED_BAZ_X = float(PRED_BAZ_X)
PRED_BAZ_Y = float(PRED_BAZ_Y)

name = str(event_time.year) + f'{event_time.month:02d}' + f'{event_time.day:02d}'+ "_" + f'{event_time.hour:02d}' + f'{event_time.minute:02d}' + f'{event_time.second:02d}'

All_Lin_arr = np.load(Res_dir + "All_Lin_%s_%s_%s_array.npy" %(Boots, fmin, fmax))
All_Noise_arr = np.load(Res_dir + "All_Noise_%s_%s_%s_array.npy" %(Boots, fmin, fmax))
All_Thresh_Peaks_arr = np.load(Res_dir + "All_Thresh_Peaks_%s_%s_%s_array.npy" %(Boots, fmin, fmax))
Lin_Mean = np.mean(All_Lin_arr, axis=0)


core_samples, labels = dbscan(X=All_Thresh_Peaks_arr,eps=epsilon,min_samples=int(Boots * MinPts))
no_clusters = np.amax(labels) + 1

## get error values and means of clusters
print(labels.shape, All_Thresh_Peaks_arr.shape)


# All_Thresh_Peaks_arr[:,0] += PRED_BAZ_X
# All_Thresh_Peaks_arr[:,1] += PRED_BAZ_Y

from cluster_utilities import cluster_utilities
cu = cluster_utilities(labels = labels, points = All_Thresh_Peaks_arr)

means_xy, means_baz_slow = cu.cluster_means()
bazs_std, slows_std = cu.cluster_std_devs()
ellipse_areas = cu.cluster_ellipse_areas(std_dev=2)

print('number of arrivals: \n', no_clusters, '\n')
print('means [slow_x,slow_y]: \n',means_xy, '\n[means baz,slow]: \n', means_baz_slow, '\n')
print('baz std dev: ',bazs_std, '\nslow std dev: ', slows_std, '\n')
print('Ellipse areas: ',ellipse_areas)


# plot!

from array_plotting import plotting
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
p = plotting(ax = ax)

x_min = PRED_BAZ_X + slow_min
x_max = PRED_BAZ_X + slow_max
y_min = PRED_BAZ_Y + slow_min
y_max = PRED_BAZ_Y + slow_max



p.plot_clusters_XY(labels=labels, tp=Lin_Mean, peaks=All_Thresh_Peaks_arr,
                   sxmin=x_min, sxmax=x_max, symin=y_min, symax=y_max,
                   sstep=s_space, title="Cluster test plot", predictions=predictions)

plt.savefig("Cluster.pdf")
plt.show()
