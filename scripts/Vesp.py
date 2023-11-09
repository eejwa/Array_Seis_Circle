#!/usr/bin/env python

import obspy
import numpy as np
import time
import scipy 
from array_info import array
from vespagram import Vespagram_Lin, Vespagram_PWS, Baz_vespagram_PWS, Baz_vespagram_Lin
from array_plotting import plotting

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from Parameters_Vesp import *

st = obspy.read(filepath)
st = st.resample(20, window='hann',no_filter=True)
a = array(st)
# get array metadata
event_time = a.eventtime()
geometry = a.geometry()
distances = a.distances(type="deg")
mean_dist = np.mean(distances)
stations = a.stations()

# get travel time information and define a window
Target_phase_times, time_header_times = a.get_predicted_times(phase)

avg_target_time = np.mean(Target_phase_times)
min_target = int(np.nanmin(Target_phase_times, axis=0)) + cut_min
max_target = int(np.nanmax(Target_phase_times, axis=0)) + cut_max
window = np.array([min_target, max_target])

stime = event_time + min_target
etime = event_time + max_target

# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()


if phase not in phases:
    phases.append(phase)

# get predicted slownesses and backazimuths
predictions = a.pred_baz_slow(phases=phases, one_eighty=True)



# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[
    row, :
][0]

# filter
st = st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

# get the traces and phase traces
array_processed = array(st)
Traces = array_processed.traces()
Phase_traces = array_processed.phase_traces()

mode_length = scipy.stats.mode([trace.shape[0] for trace in Traces])[0]
print(mode_length)

Traces = np.array([t for t in Traces if t.shape[0] == mode_length])
Phase_traces = np.array([t for t in Phase_traces if t.shape[0] == mode_length])

print(Traces.shape)
print(Phase_traces.shape)

# get sampleing rate
sampling_rate = st[0].stats.sampling_rate


if Vesp_type == "slow":
    kwarg_dict = {
        "traces": Traces,
        "sampling_rate": sampling_rate,
        "geometry": geometry,
        "distance": mean_dist,
        "baz": float(BAZ),
        "smin": smin,
        "smax": smax,
        "s_space": s_step,
    }

    if Stack_type == "Lin":

        vesp = Vespagram_Lin(**kwarg_dict)

    elif Stack_type == "PWS":
        vesp = Vespagram_PWS(**kwarg_dict, phase_traces=Phase_traces, degree=degree)

    ymin = smin
    ymax = smax
    y_step = s_step

elif Vesp_type == "baz":
    kwarg_dict = {
        "traces": Traces,
        "sampling_rate": sampling_rate,
        "geometry": geometry,
        "distance": mean_dist,
        "slow": float(S),
        "bmin": bmin,
        "bmax": bmax,
        "b_space": b_step,
    }

    if Stack_type == "Lin":

        vesp = Baz_vespagram_Lin(**kwarg_dict)

    elif Stack_type == "PWS":
        vesp = Baz_vespagram_PWS(**kwarg_dict, phase_traces=Phase_traces, degree=degree)

    ymin = bmin
    ymax = bmax
    y_step = b_step

else:
    print('Vesp_type needs to be "slow" or "baz"')
    exit()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
p = plotting(ax)


print(min_target, max_target, sampling_rate)
print((max_target - min_target)* sampling_rate)
print(Traces.shape)
p.plot_vespagram(
    vespagram=vesp,
    ymin=ymin,
    ymax=ymax,
    y_space=y_step,
    tmin=min_target,
    tmax=max_target,
    sampling_rate=sampling_rate,
    title="%s Vespagram. %s Stacking" % (Vesp_type, Stack_type),
    predictions=predictions,
    type=Vesp_type,
    envelope=True,
    normalise=True
)

plt.savefig(f'Vespagram_{phase}.pdf')
# plt.show()
