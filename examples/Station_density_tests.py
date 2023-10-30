#!/usr/bin/env python


# testing ways to get station density values

import obspy
import matplotlib.pyplot as plt
import numpy as np

from array_info import array
from make_sub_array import get_station_density_KDE
from beamforming_xy import BF_XY_Lin
from array_plotting import plotting

st = obspy.read('./data/19970525/*SAC')
a = array(st)

geometry = a.geometry()


lons = geometry[:,0]
lats = geometry[:,1]

density = get_station_density_KDE(geometry)

density = np.exp(density)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
s = ax.scatter(lons,lats,c=density, label='stations', marker='^')
plt.colorbar(s, ax=ax)
ax.set_facecolor('black')
ax.set_title('Density plot')
ax.set_xlabel('Longitude ($^{\circ}$)')
ax.set_ylabel('Latitude ($^{\circ}$)')
ax.legend(loc='best')
plt.show()


# weight data with the normalised densities
density /= density.max()

# conduct beamforming
phase = 'SKS'
phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']

# frequency band
fmin = 0.1
fmax = 0.4

# define area around predictions to do analysis
box_len = 3

# get array metadata
event_time = a.eventtime()
distances = a.distances(type='deg')
mean_dist = np.mean(distances)
stations = a.stations()

# get travel time information and define a window
Target_phase_times, time_header_times = a.get_predicted_times(phase)

avg_target_time = np.mean(Target_phase_times)
min_target_time = int(np.nanmin(Target_phase_times, axis=0))
max_target_time = int(np.nanmax(Target_phase_times, axis=0))

stime = event_time + min_target_time
etime = event_time + max_target_time + 0


# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()

# get predicted slownesses and backazimuths
predictions = a.pred_baz_slow(phases=phases, one_eighty=True)

# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]


# make the box around the prediction to search over
slow_x_min = round(float(PRED_BAZ_X) - box_len, 2)
slow_x_max = round(float(PRED_BAZ_X) + box_len, 2)
slow_y_min = round(float(PRED_BAZ_Y) - box_len, 2)
slow_y_max = round(float(PRED_BAZ_Y) + box_len, 2)
s_space = 0.05

# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)
# get the traces and phase traces
Traces = a.traces()
Phase_traces = a.phase_traces()

# get sampling rate
sampling_rate=st[0].stats.sampling_rate

# weight by station density
# for i,tr in enumerate(Traces):
#     Traces[i] /= (density[i])

# run the beamforming!
Lin_arr, Results_arr, peaks = BF_XY_Lin(traces=Traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, sxmin=slow_x_min,
                                                        sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max, s_space=s_space)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
p = plotting(ax)
p.plot_TP_XY(tp=Lin_arr, peaks=peaks, sxmin=slow_x_min, sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max,
          sstep=s_space, contour_levels=10, predictions=predictions, log=False, title='Not Weighted')

for i,tr in enumerate(Traces):
    Traces[i] /= (density[i] / density.max())

Lin_arr, Results_arr, peaks = BF_XY_Lin(traces=Traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, sxmin=slow_x_min,
                                                        sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max, s_space=s_space)

ax = fig.add_subplot(122)
p = plotting(ax)
p.plot_TP_XY(tp=Lin_arr, peaks=peaks, sxmin=slow_x_min, sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max,
          sstep=s_space, contour_levels=10, predictions=predictions, log=False, title='Station Density Weighted')

plt.tight_layout()
plt.show()
