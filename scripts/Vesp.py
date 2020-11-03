#!/usr/bin/env python

import obspy
import numpy as np
import time

from circ_array import circ_array
from circ_beam import Vespagram_Lin, Vespagram_PWS, Baz_vespagram_PWS, Baz_vespagram_Lin
from array_plotting import plotting
c = circ_array()
p = plotting()

from Parameters_Vesp import *

st = obspy.read(filepath)

# get array metadata
event_time = c.get_eventtime(st)
geometry = c.get_geometry(st)
distances = c.get_distances(st,type='deg')
mean_dist = np.mean(distances)
stations = c.get_stations(st)

# get travel time information and define a window
Target_phase_times, time_header_times = c.get_predicted_times(st,phase)

avg_target_time = np.mean(Target_phase_times)
min_target = int(np.nanmin(Target_phase_times, axis=0)) - cut_min
max_target = int(np.nanmax(Target_phase_times, axis=0)) + cut_max
window = np.array([min_target, max_target])

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

# get sampleing rate
sampling_rate=st[0].stats.sampling_rate


if Vesp_type == 'slow':
    kwarg_dict = {'traces':Traces, 'sampling_rate':sampling_rate, 'geometry':geometry,
                             'distance':mean_dist, 'baz':float(BAZ), 'smin':smin, 'smax':smax, 's_space':s_step}

    if Stack_type == 'Lin':

        vesp = Vespagram_Lin(**kwarg_dict)

    elif Stack_type == 'PWS':
        vesp = Vespagram_PWS(**kwarg_dict, phase_traces=Phase_traces, degree=degree)

    ymin = smin
    ymax = smax
    y_step = s_step
elif Vesp_type == 'baz':
    kwarg_dict = {'traces':Traces, 'sampling_rate':sampling_rate, 'geometry':geometry,
                             'distance':mean_dist, 'slow':float(S), 'bmin':bmin, 'bmax':bmax, 'b_space':b_step}

    if Stack_type == 'Lin':

        vesp = Baz_vespagram_Lin(**kwarg_dict)


    elif Stack_type == 'PWS':
        vesp = Baz_vespagram_PWS(**kwarg_dict, phase_traces=Phase_traces, degree=degree)

    ymin = bmin
    ymax = bmax
    y_step = b_step

else:
    print('Vesp_type needs to be "slow" or "baz"')
    exit()

p.plot_vespagram(vespagram=vesp, ymin=ymin, ymax=ymax, y_space=y_step, tmin=min_target, tmax=max_target,
                 sampling_rate=sampling_rate, title="%s Vespagram. %s Stacking" %(Vesp_type, Stack_type),
                 predictions=predictions, type='slow', envelope=True)
