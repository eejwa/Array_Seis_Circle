#!/usr/bin/env python

# Using my package, this will ask the user to pick a time window and then
# conduct the beamforming in this window

# imports
import obspy
import numpy as np
import time
from obspy.taup import TauPyModel
from manual_pick import pick_tw

from output_writing import write_to_file
from array_info import array as a
from beamforming_polar import BF_Pol_all, BF_Pol_Lin, BF_Pol_PWS
from shift_stack import shift_traces
from array_plotting import plotting
import matplotlib.pyplot as plt

# import parameters
from Parameters_TP_Pol import *
taup = TauPyModel(model=model)

st = obspy.read(filepath)
array = a(st)
# get array metadata
event_time = array.eventtime()
geometry = array.geometry()
distances = array.distances(type="deg")
mean_dist = np.mean(distances)
stations = array.stations()
centre_x, centre_y = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
sampling_rate = st[0].stats.sampling_rate
evdp = st[0].stats.sac.evdp

# convert elevation to km
geometry[:,2] /= 1000

# filter
if Filter == True:
    st = st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
else:
    st = st.copy()

# get travel time information and define a window
if phase is not None:
    Target_phase_times, time_header_times = array.get_predicted_times(phase=phase)

    min_target = int(np.nanmin(Target_phase_times, axis=0)) + cut_min
    max_target = int(np.nanmax(Target_phase_times, axis=0)) + cut_max

    stime = event_time + min_target
    etime = event_time + max_target
    #
    # # trim the stream
    # Normalise and cut seismogram around defined window
    st_trim = st.copy().trim(starttime=stime, endtime=etime)
    st_norm = st_trim.normalize()
else:
    st_trim = st.copy()#.trim(starttime=stime, endtime=etime)
    st_norm = st_trim.normalize()

# get predicted slownesses and backazimuths
predictions = array.pred_baz_slow(phases=phases, one_eighty=True)

# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[
    row, :
][0]

if Man_Pick == True:

    # get the user to pick the time window
    window = pick_tw(stream=st, phase=phase, align=Align)

    rel_tmin = window[0]
    rel_tmax = window[1]

elif Man_Pick == False:
    if Align == False:
        rel_tmin = t_min + int(np.nanmin(Target_phase_times, axis=0))
        rel_tmax = t_max + int(np.nanmax(Target_phase_times, axis=0))
        window = np.array([t_min, t_max])

    if Align == True:
        rel_tmin = t_min
        rel_tmax = t_max
        window = np.array([t_min, t_max])

else:
    print("Man_Pick needs to be set to True or False!")
    exit()

if Align == True:

    array_norm = a(st_norm)
    # get the traces and phase traces
    Traces = array_norm.traces()
    Phase_traces = array_norm.phase_traces()

    # align the traces and phase traces
    Shifted_Traces = shift_traces(
        traces=Traces,
        geometry=geometry,
        abs_slow=float(S),
        baz=float(BAZ),
        distance=float(mean_dist),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
        elevation=True,
        incidence=8
    )

    Shifted_Phase_Traces = shift_traces(
        traces=Phase_traces,
        geometry=geometry,
        abs_slow=float(S),
        baz=float(BAZ),
        distance=float(mean_dist),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
        elevation=True,
        incidence=8
    )

    # uncomment to check the shifting has worked
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111)
    # for b,trace in enumerate(Shifted_Traces):
    #     ax.plot(trace + distances[b], color='black')
    #
    # plt.show()
    # exit()

    ## cut the shifted traces within the defined time window
    ## from the rel_tmin and rel_tmax

    arrivals = taup.get_travel_times(
        source_depth_in_km=evdp, distance_in_degree=mean_dist, phase_list=[phase]
    )

    pred_point = int(sampling_rate * (arrivals[0].time - min_target))
    point_before = int(pred_point + (rel_tmin * sampling_rate))
    point_after = int(pred_point + (rel_tmax * sampling_rate))

    cut_traces = Shifted_Traces[:, point_before:point_after]
    cut_phase_traces = Shifted_Phase_Traces[:, point_before:point_after]

    s_min = 0
    s_max = slow_max
    b_min = 0
    b_max = 360

elif Align == False:
    stime = event_time + rel_tmin
    etime = event_time + rel_tmax

    # trim the stream
    # Normalise and cut seismogram around defined window

    st_trim = st.copy().trim(starttime=stime, endtime=etime)
    st_norm = st_trim.normalize()

    array_norm = a(st_norm)
    # get the traces and phase traces
    cut_traces = array_norm.traces()
    cut_phase_traces = array_norm.phase_traces()

    s_min = float(S) + slow_min
    s_max = float(S) + slow_max
    b_min = float(BAZ) + baz_min
    b_max = float(BAZ) + baz_max


# define kwarg dictionary

kwarg_dict = {
    "traces": cut_traces,
    "sampling_rate": np.float64(sampling_rate),
    "geometry": geometry,
    "distance": mean_dist,
    "smin": s_min,
    "smax": s_max,
    "bazmin": b_min,
    "bazmax": b_max,
    "s_space": s_space,
    "baz_space": b_space,
    "elevation":True,
    "incidence":8
}

# uncomment to check traces aligned and cut properly
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)
# for b,trace in enumerate(cut_traces):
#     ax.plot(trace + distances[b], color='black')
#
# plt.show()


if Stack_type == "Both":
    # run the beamforming!

    Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_Pol_all(
        phase_traces=cut_phase_traces, degree=degree, **kwarg_dict
    )


    peaks = np.c_[peaks, np.array(["PWS", "LIN", "F"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]


elif Stack_type == "Lin" or Stack_type == "LIN":

    Lin_arr, Results_arr, peaks = BF_Pol_Lin(**kwarg_dict)
    peaks = np.c_[peaks, np.array(["LIN"])]

    Plot_arr = Lin_arr / Lin_arr.max()

    peaks = peaks[np.where(peaks == "LIN")[0]]

elif Stack_type == "PWS":

    PWS_arr, Results_arr, peaks = BF_Pol_PWS(
        phase_traces=cut_phase_traces, degree=degree, **kwarg_dict
    )

    peaks = np.c_[peaks, np.array(["PWS"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]

# write to file

if Align == True:
    peaks[:, 0] = float(peaks[:, 0]) + float(BAZ)
    peaks[:, 1] = float(peaks[:, 1]) + float(S)

## write to file
filepath = Res_dir + "Pol_Results.txt"


slow_vec_obs = np.array(peaks[:, :2]).astype(float)
pred_file = np.array([BAZ, S]).astype(float)


write_to_file(
    filepath=filepath,
    st=st,
    peaks=slow_vec_obs,
    prediction=pred_file,
    phase=phase,
    time_window=window,
)

# s_min_plot = float(S) + slow_min
# s_max_plot = float(S) + slow_max
# b_min_plot = float(BAZ) + baz_min
# b_max_plot = float(BAZ) + baz_max

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="polar")
p = plotting(ax)
p.plot_TP_Pol(
    tp=Plot_arr,
    peaks=peaks,
    smin=s_min,
    smax=s_max,
    bazmin=b_min,
    bazmax=b_max,
    sstep=s_space,
    bazstep=b_space,
    contour_levels=20,
    title="%s Plot" % Stack_type,
    predictions=predictions,
    log=False,
)

plt.show()
