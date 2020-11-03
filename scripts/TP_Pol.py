#!/usr/bin/env python

# Using my package, this will ask the user to pick a time window and then
# conduct the beamforming in this window

# imports
import obspy
import numpy as np
import time
from obspy.taup import TauPyModel
model = TauPyModel(model="prem")

from circ_array import circ_array
from circ_beam import BF_Spherical_Pol_all, BF_Spherical_Pol_Lin, BF_Spherical_Pol_PWS, shift_traces
from array_plotting import plotting
c = circ_array()


# import parameters
from Parameters_TP_Pol import *

st = obspy.read(filepath)

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

print(predictions)
# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]


if Man_Pick == True:

    # get the user to pick the time window
    window = c.pick_tw(stream=st, phase=phase, align=Align)

    rel_tmin = window[0]
    rel_tmax = window[1]
elif Man_Pick == False:
    rel_tmin = t_min
    rel_tmax = t_max
    window = np.array([t_min, t_max])
else:
    print("Man_Pick needs to be set to True or False!")
    exit()

# filter
st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                  corners=4, zerophase=True)


if Align == True:

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
    point_before= int(pred_point + (rel_tmin * sampling_rate))
    point_after= int(pred_point + (rel_tmax * sampling_rate))

    cut_traces = Shifted_Traces[:,point_before:point_after]
    cut_phase_traces = Shifted_Phase_Traces[:,point_before:point_after]

    s_min = 0
    s_max = float(S) + slow_max
    b_min = 0
    b_max = 360

elif Align == False:
    stime = event_time + rel_tmin
    etime = event_time + rel_tmax

    # trim the stream
    # Normalise and cut seismogram around defined window
    st = st.copy().trim(starttime=stime, endtime=etime)
    st = st.normalize()

    # get the traces and phase traces
    cut_traces = c.get_traces(st)
    cut_phase_traces = c.get_phase_traces(st)

    s_min = float(S) + slow_min
    s_max = float(S) + slow_max
    b_min = float(BAZ) + baz_min
    b_max = float(BAZ) + baz_max


# define kwarg dictionary

kwarg_dict = {'traces':cut_traces,  'sampling_rate':np.float64(sampling_rate),
        'geometry':geometry, 'distance':mean_dist,
        'smin':s_min, 'smax':s_max, 'bazmin':b_min,
        'bazmax':b_max, 's_space':s_space, 'baz_space':b_space}


if Stack_type == 'Both':
    # run the beamforming!
    start = time.time()
    Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_Spherical_Pol_all(phase_traces=cut_phase_traces, degree=2, **kwarg_dict)
    end = time.time()

    peaks = np.c_[peaks, np.array(["PWS", "LIN", "F"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]


elif Stack_type == 'Lin':

    start = time.time()
    Lin_arr, Results_arr, peaks = BF_Spherical_Pol_Lin(**kwarg_dict)
    end = time.time()

    peaks = np.c_[peaks, np.array(["LIN"])]
    Plot_arr = Lin_arr / Lin_arr.max()

    peaks = peaks[np.where(peaks == "LIN")[0]]

elif Stack_type == 'PWS':

    start = time.time()
    PWS_arr, Results_arr, peaks = BF_Spherical_Pol_PWS(phase_traces=cut_phase_traces, degree=2, **kwarg_dict)
    end = time.time()

    peaks = np.c_[peaks, np.array(["PWS"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]

# write to file

if Align == True:
    peaks[:,0] = float(peaks[:,0]) + float(BAZ)
    peaks[:,1] = float(peaks[:,1]) + float(S)

## write to file
filepath = Res_dir + "Pol_Results.txt"


slow_vec_obs = np.array(peaks[:,:2]).astype(float)
pred_file = np.array([BAZ,S]).astype(float)
print(slow_vec_obs)

c.write_to_file(filepath=filepath, st=st, peaks=slow_vec_obs, prediction=pred_file, phase=phase, time_window=window)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='polar')
p = plotting(ax)
p.plot_TP_Pol(tp=Plot_arr, peaks=peaks, smin=s_min, smax=s_max,
              bazmin=b_min, bazmax=b_max,
              sstep=s_space, bazstep=b_space, contour_levels=20,
              title="%s Plot" %Stack_type, predictions=predictions, log=False)

plt.show()
