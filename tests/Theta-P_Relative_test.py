#!/usr/bin/env python
# file to test out the relative theta - p codes!
# In this case, you align the traces on the predicted backazimuth and slowness
# then conduct the beamforming

# imports
import obspy
import numpy as np
import time
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
model = TauPyModel(model="prem")

import circ_array as c
from circ_beam import BF_Spherical_XY_all, shift_traces, BF_Spherical_XY_PWS, BF_Spherical_XY_Lin
from array_plotting import plotting



# parameters
# phase of interest
phase = 'SKS'
phases = ['SKS', 'SKKS', 'ScS', 'Sdiff', 'sSKS', 'pSKS', 'sSKKS', 'PS']

# frequency band
fmin = 0.15
fmax = 0.60

rel_tmin = 20 # to be subtracted from shifted trace times
rel_tmax = 30 # to be added to shifted trace times

# shift traces doesnt work unless the stream is cut
# this is just to initially cut the traces around
# the predicted arrival time.
cut_min = 50
cut_max = 50

# define area around predictions to do analysis

st = obspy.read('./data/19970521/*SAC')

# get array metadata
event_time = c.get_eventtime(st)
geometry = c.get_geometry(st)
distances = c.get_distances(st,type='deg')
mean_dist = np.mean(distances)
stations = c.get_stations(st)
centre_x, centre_y =  np.mean(geometry[:, 0]),  np.mean(geometry[:, 1])
sampling_rate=st[0].stats.sampling_rate
evdp = st[0].stats.sac.evdp

# get travel time information and define a window
Target_phase_times, time_header_times = c.get_predicted_times(st,phase)

min_target = int(np.nanmin(Target_phase_times, axis=0)) - cut_min
max_target = int(np.nanmax(Target_phase_times, axis=0)) + cut_max

stime = event_time + min_target
etime = event_time + max_target

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



# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)

# get the traces and phase traces
Traces = c.get_traces(st)
Phase_traces = c.get_phase_traces(st)


# align the traces and phase traces
Shifted_Traces = shift_traces(traces=Traces, geometry=geometry, abs_slow=float(S), baz=float(BAZ),
                              distance=float(mean_dist), centre_x=float(centre_x), centre_y=float(centre_y),
                              sampling_rate=sampling_rate)

Shifted_Phase_Traces = shift_traces(traces=Phase_traces, geometry=geometry, abs_slow=float(S), baz=float(BAZ),
                                 distance=float(mean_dist), centre_x=float(centre_x), centre_y=float(centre_y),
                                 sampling_rate=sampling_rate)

## cut the shifted traces within the defined time window
## from the rel_tmin and rel_tmax

arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=mean_dist, phase_list=[phase])

pred_point = int(sampling_rate * (arrivals[0].time - min_target))
point_before= int(pred_point - (rel_tmin * sampling_rate))
point_after= int(pred_point + (rel_tmax * sampling_rate))

cut_shifted_traces = Shifted_Traces[:,point_before:point_after]
cut_shifted_phase_traces = Shifted_Phase_Traces[:,point_before:point_after]

# define slowness box
slow_min = -3
slow_max = 3
s_space = 0.05

# run the beamforming!
start = time.time()
Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_Spherical_XY_all(traces=cut_shifted_traces, phase_traces=cut_shifted_phase_traces,
                                                        sampling_rate=np.float64(sampling_rate), geometry=geometry, distance=mean_dist,
                                                        sxmin=slow_min,sxmax=slow_max, symin=slow_min, symax=slow_max,
                                                        s_space=s_space, degree=4)
end = time.time()
print("time to run:", end - start)

peaks = np.c_[peaks, np.array(["PWS", "LIN", "F"])]

peaks = peaks[np.where(peaks == "PWS")[0]]


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
p = plotting(ax = ax)

p.plot_TP_XY(tp=PWS_arr, peaks=peaks, sxmin=slow_min, sxmax=slow_max, symin=slow_min, symax=slow_max,
          sstep=s_space, contour_levels=50, title="PWS Plot", predictions=None, log=False)

plt.show()

start = time.time()
Lin_arr, Results_arr, peaks = BF_Spherical_XY_Lin(traces=cut_shifted_traces, sampling_rate=np.float64(sampling_rate),
                                                        geometry=geometry, distance=mean_dist,
                                                        sxmin=slow_min,sxmax=slow_max, symin=slow_min,
                                                        symax=slow_max, s_space=s_space)
end = time.time()
print("time to run:", end - start)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
p = plotting(ax = ax)

p.plot_TP_XY(tp=Lin_arr, peaks=peaks, sxmin=slow_min, sxmax=slow_max, symin=slow_min, symax=slow_max,
          sstep=s_space, contour_levels=50, title="LIN Plot", predictions=None, log=False)
plt.show()

start = time.time()
PWS_arr, Results_arr, peaks = BF_Spherical_XY_PWS(traces=cut_shifted_traces, phase_traces=cut_shifted_phase_traces,
                                                  sampling_rate=np.float64(sampling_rate), geometry=geometry, distance=mean_dist,
                                                  sxmin=slow_min,sxmax=slow_max, symin=slow_min, symax=slow_max,
                                                  s_space=s_space, degree=4)
end = time.time()
print("time to run:", end - start)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
p = plotting(ax = ax)

p.plot_TP_XY(tp=PWS_arr, peaks=peaks, sxmin=slow_min, sxmax=slow_max, symin=slow_min, symax=slow_max,
          sstep=s_space, contour_levels=50, title="PWS Plot", predictions=None, log=False)
plt.show()
