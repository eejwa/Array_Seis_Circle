#!/usr/bin/env python

# file to test out the theta - p codes and plotting!

# imports
import obspy
import numpy as np
import time
import matplotlib.pyplot as plt

from circ_array import circ_array
from circ_beam import BF_Spherical_Pol_all, BF_Spherical_Pol_PWS, BF_Spherical_Pol_Lin
from array_plotting import plotting
c = circ_array()

# parameters
# phase of interest
phase = 'SKS'
phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']

# frequency band
fmin = 0.15
fmax = 0.60

st = obspy.read('./data/19970521/*SAC')

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

slow_min = float(S) - 2
slow_max = float(S) + 2
baz_min = float(BAZ) - 30
baz_max = float(BAZ) + 30
b_space = 1
s_space = 0.1

# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)

# get the traces and phase traces
Traces = c.get_traces(st)
Phase_traces = c.get_phase_traces(st)

# get sampleing rate
sampling_rate=st[0].stats.sampling_rate


start = time.time()
Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_Spherical_Pol_all(traces=Traces, phase_traces=Phase_traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, smin=slow_min,
                                                        smax=slow_max, bazmin=baz_min, bazmax=baz_max, s_space=0.1, baz_space=b_space, degree=2)
end = time.time()

print("time", end-start)
peaks = np.c_[peaks, np.array(["PWS", "LIN", "F"])]

peaks = peaks[np.where(peaks == "PWS")[0]]


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='polar')
p = plotting(ax = ax)

p.plot_TP_Pol(tp=PWS_arr, peaks=peaks, smin=slow_min, smax=slow_max, bazmin=baz_min, bazmax=baz_max,
              sstep=s_space, bazstep=b_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)
plt.show()






start = time.time()
Lin_arr, Results_arr, peaks = BF_Spherical_Pol_Lin(traces=Traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, smin=slow_min,
                                                        smax=slow_max, bazmin=baz_min, bazmax=baz_max, s_space=s_space, baz_space=b_space)
end = time.time()

print("time", end-start)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='polar')
p = plotting(ax = ax)

p.plot_TP_Pol(tp=Lin_arr, peaks=peaks, smin=slow_min, smax=slow_max, bazmin=baz_min, bazmax=baz_max,
              sstep=s_space, bazstep=b_space, contour_levels=50, title="Lin Plot", predictions=predictions, log=False)

plt.show()




start = time.time()
PWS_arr, Results_arr, peaks = BF_Spherical_Pol_PWS(traces=Traces, phase_traces=Phase_traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, smin=slow_min,
                                                        smax=slow_max, bazmin=baz_min, bazmax=baz_max, s_space=s_space, baz_space=b_space, degree=2)
end = time.time()

print("time", end-start)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='polar')
p = plotting(ax = ax)

p.plot_TP_Pol(tp=PWS_arr, peaks=peaks, smin=slow_min, smax=slow_max, bazmin=baz_min, bazmax=baz_max,
              sstep=s_space, bazstep=b_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)
plt.show()
