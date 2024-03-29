#!/usr/bin/env python

Description = """
This python script will test:
    - Calculating the $\theta - p$ plot with a noise estimate.
    - Using the noise estimate as a threshold filter.
"""

from array_info import array
from beamforming_xy import BF_Noise_Threshold_Relative_XY
from shift_stack import shift_traces
from extract_peaks import findpeaks_XY

import obspy
import matplotlib.pyplot as plt
import numpy as np
import scipy

from obspy.taup import TauPyModel
model = TauPyModel(model="prem")

from array_plotting import plotting


# parameters
# phase of interest
target_phase = 'SKS'
phases = [target_phase,'SKKS','ScS','Sdiff','sSKS','sSKKS','PS']

# frequency band
fmin = 0.15
fmax = 0.60
rel_tmin = -20 # to be subtracted from shifted trace times
rel_tmax = 30 # to be added to shifted trace times

# shift traces doesnt work unless the stream is cut
# this is just to initially cut the traces around
# the predicted arrival time.
cut_min = -50
cut_max = 50

# define area around predictions to do analysis

st = obspy.read('./data/19970525/*SAC')
a = array(st)
# get array metadata
event_time = a.eventtime()
geometry = a.geometry()
distances = a.distances(type='deg')
mean_dist = np.mean(distances)
stations = a.stations()
centre_x, centre_y =  np.mean(geometry[:, 0]),  np.mean(geometry[:, 1])
sampling_rate=st[0].stats.sampling_rate
evdp = st[0].stats.sac.evdp


# get travel time information and define a window
Target_phase_times, time_header_times = a.get_predicted_times(target_phase)

min_target = int(np.nanmin(Target_phase_times, axis=0)) + cut_min
max_target = int(np.nanmax(Target_phase_times, axis=0)) + cut_max

stime = event_time + min_target
etime = event_time + max_target

# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()

# get predicted slownesses and backazimuths
predictions = a.pred_baz_slow(phases=phases, one_eighty=True)

# find the line with the predictions for the phase of interest
row = np.where((predictions == target_phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]


# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)
a_processed = array(st)
# get the traces and phase traces
Traces = a_processed.traces()


# align the traces and phase traces
Shifted_Traces = shift_traces(traces=Traces, geometry=geometry, abs_slow=float(S), baz=float(BAZ),
                              distance=float(mean_dist), centre_x=float(centre_x), centre_y=float(centre_y),
                              sampling_rate=sampling_rate)

## cut the shifted traces within the defined time window
## from the rel_tmin and rel_tmax

arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=mean_dist, phase_list=[target_phase])

pred_point = int(sampling_rate * (arrivals[0].time - min_target))
point_before= int(pred_point + (rel_tmin * sampling_rate))
point_after= int(pred_point + (rel_tmax * sampling_rate))

cut_shifted_traces = Shifted_Traces[:,point_before:point_after]

# define slowness box
# coarse box for the sake of speed
slow_min = -2
slow_max = 2
s_space = 0.1


lin_tp, noise_mean, max_peak =  BF_Noise_Threshold_Relative_XY(traces=cut_shifted_traces,
                                                               sampling_rate=sampling_rate,
                                                               geometry=geometry, distance=mean_dist,
                                                               sxmin=slow_min, sxmax=slow_max,
                                                               symin=slow_min, symax=slow_max,
                                                               s_space=s_space)
Threshold_lin_array = np.copy(lin_tp)
Threshold_lin_array[lin_tp <=noise_mean * 3] = 0

Smoothed_thresh_lin_array = scipy.ndimage.filters.gaussian_filter(
                            Threshold_lin_array, 1, mode='constant')

peaks, powers = findpeaks_XY(Smoothed_thresh_lin_array, xmin=slow_min,
                       xmax=slow_max, ymin=slow_min, ymax=slow_max,
                       xstep=s_space, ystep=s_space, N=3)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
p = plotting(ax = ax)

p.plot_TP_XY(tp=Smoothed_thresh_lin_array, peaks=peaks, sxmin=slow_min, sxmax=slow_max,
             symin=slow_min, symax=slow_max, sstep=s_space, contour_levels=20,
             title="Smoothed threshold $\\theta - p$ plot", predictions=None, log=False)

plt.show()
