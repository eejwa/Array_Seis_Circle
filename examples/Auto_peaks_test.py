#!/usr/bin/env python

# script to test/show the effectiveness of the automatic peak detector function


import obspy
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt

from array_info import array 
from beamforming_xy import BF_XY_PWS
from beamforming_polar import  BF_Pol_PWS
from array_plotting import plotting
from extract_peaks import findpeaks_XY, findpeaks_Pol
from shift_stack import shift_traces
from obspy.taup import TauPyModel
model = TauPyModel(model="prem")


# first we need to get data and make a Theta-P plot:

# parameters
# phase of interest
phase = 'SKS'
phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']

# frequency band
fmin = 0.13
fmax = 0.52

rel_tmin = -20 # to be subtracted from shifted trace times
rel_tmax = 30 # to be added to shifted trace times

# define area around predictions to do analysis
box_len = 3

st = obspy.read('./data/19970525/*SAC')

a = array(st)
# get array metadata
event_time = a.eventtime()
geometry = a.geometry()
distances = a.distances(type='deg')
mean_dist = np.mean(distances)
stations = a.stations()
# get sampleing rate
sampling_rate=st[0].stats.sampling_rate
centre_x, centre_y =  np.mean(geometry[:, 0]),  np.mean(geometry[:, 1])
evdp = st[0].stats.sac.evdp

# get travel time information and define a window
Target_phase_times, time_header_times = a.get_predicted_times(phase)

avg_target_time = np.mean(Target_phase_times)
min_target= int(np.nanmin(Target_phase_times, axis=0)) - 50 
max_target = int(np.nanmax(Target_phase_times, axis=0)) + 50

stime = event_time + min_target
etime = event_time + max_target 

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
slow_x_min = -2
slow_x_max = 2
slow_y_min = -2
slow_y_max = 2
s_space = 0.05

# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)

# get the traces and phase traces
a_processed = array(st)
Traces = a_processed.traces()
Traces = a_processed.traces()
Phase_traces = a_processed.phase_traces()

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
point_before= int(pred_point + (rel_tmin * sampling_rate))
point_after= int(pred_point + (rel_tmax * sampling_rate))

cut_shifted_traces = Shifted_Traces[:,point_before:point_after]
cut_shifted_phase_traces = Shifted_Phase_Traces[:,point_before:point_after]

print(Traces.shape)
print(point_before, point_after, pred_point, min_target)
print(Shifted_Traces.shape)
#run the beamforming!

PWS_arr, Results_arr, peaks = BF_XY_PWS(traces=cut_shifted_traces, phase_traces=cut_shifted_phase_traces, sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, sxmin=slow_x_min,
                                                        sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max, s_space=s_space, degree=2)


# smooth the linear array to avoid local maxima
smoothed_arr = scipy.ndimage.filters.gaussian_filter(
    PWS_arr, 2, mode='constant')


# Ok! Now find the top 2 peaks using the findpeaks function.
peaks_auto, peaks_powers = findpeaks_XY(Array=smoothed_arr, xmin=slow_x_min, xmax=slow_x_max, 
                                        ymin=slow_y_min, ymax=slow_y_max, xstep=s_space, ystep=s_space, N=3)
print('peaks:\n',peaks_auto)


fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
p = plotting(ax)

# plot to see if they're any good.
p.plot_TP_XY(tp=PWS_arr, peaks=peaks_auto, sxmin=slow_x_min, sxmax=slow_x_max, symin=slow_y_min, symax=slow_y_max,
          sstep=s_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)

slow_min = 0
slow_max = 2
baz_min = 0
baz_max = 360
b_space = 1

PWS_arr, Results_arr, peaks = BF_Pol_PWS(traces=cut_shifted_traces, phase_traces = cut_shifted_phase_traces,sampling_rate=np.float64(
                                                        sampling_rate), geometry=geometry, distance=mean_dist, smin=slow_min,
                                                        smax=slow_max, bazmin=baz_min, bazmax=baz_max, s_space=s_space, baz_space=b_space, degree=2)

# smooth the linear array to avoid local maxima
smoothed_arr = scipy.ndimage.filters.gaussian_filter(
    PWS_arr, 2, mode='constant')

peaks_auto, peaks_powers = findpeaks_Pol(Array=smoothed_arr, bmin=baz_min, bmax=baz_max, smin=slow_min, smax=slow_max, sstep=s_space, bstep=b_space, N=3)
print('peaks:\n',peaks_auto)

ax2 = fig.add_subplot(122, projection='polar')
p = plotting(ax2)
p.plot_TP_Pol(tp=PWS_arr, peaks=peaks_auto, smin=slow_min, smax=slow_max, bazmin=baz_min, bazmax=baz_max,
              sstep=s_space, bazstep=b_space, contour_levels=50, title="PWS Plot", predictions=predictions, log=False)

plt.tight_layout()
plt.show()
