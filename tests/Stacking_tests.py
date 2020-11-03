#!/usr/bin/env python

# testing code for:
#   - The stacking of traces over a given backazimuth and slowness

import obspy
import numpy as np
import matplotlib.pyplot as plt

from circ_array import circ_array
from circ_beam import pws_stack_baz_slow, linear_stack_baz_slow
c = circ_array()

phase = 'SKS'
phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']


st = obspy.read('./data/19970525/*SAC')

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

print(stime)
print(etime)

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



# get the traces and phase traces
Traces = c.get_traces(st)
Phase_traces = c.get_phase_traces(st)

# get sampleing rate
sampling_rate=st[0].stats.sampling_rate


#stack!

lin_stack = linear_stack_baz_slow(traces=Traces, sampling_rate=sampling_rate, geometry=geometry,
                                  distance=mean_dist, slow=float(S), baz=float(BAZ))

pws_stack = pws_stack_baz_slow(traces=Traces, phase_traces=Phase_traces, sampling_rate=sampling_rate,
                               geometry=geometry, distance=mean_dist, slow=float(S), baz=float(BAZ),
                               degree=3)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax1.plot(lin_stack)

ax2 = fig.add_subplot(212)
ax2.plot(pws_stack)

plt.show()
