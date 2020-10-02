#!/usr/bin/env python


# testing ways to get station density values

import obspy
import matplotlib.pyplot as plt
import numpy as np

from Circ_Array import Circ_Array
c = Circ_Array()
from Circ_Beam import BF_Spherical_XY_all, BF_Spherical_Pol_all
from Array_Plotting import Plotting
p = Plotting()

st = obspy.read('./data/19980329/*SAC')
geometry = c.get_geometry(st)


lons = geometry[:,0]
lats = geometry[:,1]

density = c.get_station_density_KDE(geometry)

density = np.exp(density)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
s = ax.scatter(lons,lats,c=density)
plt.colorbar(s)
ax.set_facecolor('black')
plt.show()


# weight data with the normalised densities
density /= density.max()

# conduct beamforming
phase = 'SKS'
phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']

# frequency band
fmin = 0.10
fmax = 0.40

# define area around predictions to do analysis
box_len = 3

# get array metadata
event_time = c.get_eventtime(st)
geometry = c.get_geometry(st)
distances = c.get_distances(st,type='deg')
mean_dist = np.mean(distances)
stations = c.get_stations(st)


# get travel time information and define a window
Target_phase_times, time_header_times = c.get_predicted_times(st,phase)

avg_target_time = np.mean(Target_phase_times)
min_target_time = int(np.nanmin(Target_phase_times, axis=0))
max_target_time = int(np.nanmax(Target_phase_times, axis=0))

stime = event_time + min_target_time
etime = event_time + max_target_time + 30


# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()

# get predicted slownesses and backazimuths
predictions = c.pred_baz_slow(
    stream=st, phases=phases, one_eighty=True)

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
Traces = c.get_traces(st)
Phase_traces = c.get_phase_traces(st)

# get sampleing rate
sampling_rate=st[0].stats.sampling_rate


for i,tr in enumerate(Traces):
    Traces[i] /= (density[i])

# Traces = c.get_traces(st)

# run the beamforming!

Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_Spherical_XY_all(traces=Traces, phase_traces=Phase_traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, sxmin=slow_x_min,
                                                        sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max, s_space=s_space, degree=2)

p.plot_TP_XY(tp=Lin_arr, peaks=peaks, sxmin=slow_x_min, sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max,
          sstep=s_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)

p.plot_TP_XY(tp=PWS_arr, peaks=peaks, sxmin=slow_x_min, sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max,
          sstep=s_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)
