#!/usr/bin/env python

# Using my package, this will ask the user to pick a time window and then
# conduct the beamforming in this window

# imports
# Force matplotlib to not use any Xwindows backend.
import matplotlib
# matplotlib.use('Agg')
import obspy
import numpy as np
import time
from Parameters_TP_XY import *
from obspy.taup import TauPyModel
from output_writing import write_to_file
from slow_vec_calcs import get_slow_baz
model = TauPyModel(model=pred_model)

from array_info import array
from beamforming_xy import (
    BF_XY_all,
    BF_XY_Lin,
    BF_XY_PWS
)
from shift_stack import shift_traces
from array_plotting import plotting
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
from manual_pick import pick_tw


# import parameters

st = obspy.read(filepath)
a = array(st)


# get array metadata
event_time = a.eventtime()
geometry = a.geometry()
distances = a.distances(type="deg")
mean_dist = np.mean(distances)
stations = a.stations()
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
    Target_phase_times, time_header_times = a.get_predicted_times(phase)

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
predictions = a.pred_baz_slow(phases=phases, one_eighty=True)

# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[
    row, :
][0]


if Man_Pick == True:

    # get the user to pick the time window
    window = pick_tw(stream=st, phase=phase, align=Align, tmin=cut_min, tmax=cut_max)

    rel_tmin = window[0]
    rel_tmax = window[1]

elif Man_Pick == False:
    if Align == True:
        rel_tmin = t_min
        rel_tmax = t_max
    else:
        try:
            rel_tmin = t_min + np.min(TIME)
            rel_tmax = t_max + np.max(TIME)
        except:
            rel_tmin = t_min + float(TIME)
            rel_tmax = t_max + float(TIME)
    window = np.array([t_min, t_max])
else:
    print("Man_Pick needs to be set to True or False!")
    exit()

# st_filt = st_filt.resample(20)
if Align == True:
    # get the traces and phase traces
    array_norm = array(st_norm)
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

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111)
    # for b,trace in enumerate(Shifted_Traces):
    #     ax.plot(trace + distances[b], color='black')

    # plt.show()

# nhs number 4008473149
# excess code/password: 
# - ppv argument, not up to date with literature.
# - says not earth like because of this which is wrong
# - maybe put in the paper itself 
# - email everyone mc2 related about leaving. 

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

    ## cut the shifted traces within the defined time window
    ## from the rel_tmin and rel_tmax

    arrivals = model.get_travel_times(
        source_depth_in_km=evdp, distance_in_degree=mean_dist, phase_list=[phase]
    )

    pred_point = int(sampling_rate * (arrivals[0].time - min_target))
    point_before = int(pred_point + (rel_tmin * sampling_rate))
    point_after = int(pred_point + (rel_tmax * sampling_rate))

    cut_traces = Shifted_Traces[:, point_before:point_after]
    cut_phase_traces = Shifted_Phase_Traces[:, point_before:point_after]

    sx_min = slow_min
    sx_max = slow_max
    sy_min = slow_min
    sy_max = slow_max

elif Align == False:
    stime = event_time + rel_tmin
    etime = event_time + rel_tmax

    # trim the stream
    # Normalise and cut seismogram around defined window
    st_trim = st.copy().trim(starttime=stime, endtime=etime)
    st_norm = st_trim.normalize()

    array_norm = array(st_norm)
    # get the traces and phase traces
    cut_traces = array_norm.traces()
    cut_phase_traces = array_norm.phase_traces()

    sx_min = float(PRED_BAZ_X) + slow_min
    sx_max = float(PRED_BAZ_X) + slow_max
    sy_min = float(PRED_BAZ_Y) + slow_min
    sy_max = float(PRED_BAZ_Y) + slow_max


# define kwarg dictionary

kwarg_dict = {
    "traces": cut_traces,
    "sampling_rate": np.float64(sampling_rate),
    "geometry": geometry,
    "distance": mean_dist,
    "sxmin": sx_min,
    "sxmax": sx_max,
    "symin": sy_min,
    "symax": sy_max,
    "s_space": s_space,
    "elevation":True,
    "incidence":8
}



if Stack_type == "Both":
    # run the beamforming!
    Lin_arr, PWS_arr, F_arr, Results_arr, peaks = BF_XY_all(
        phase_traces=cut_phase_traces, degree=2, **kwarg_dict
    )

    peaks = np.c_[peaks, np.array(["PWS", "LIN", "F"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]


elif Stack_type == "LIN":

    Lin_arr, Results_arr, peaks = BF_XY_Lin(**kwarg_dict)


    peaks = np.c_[peaks, np.array(["LIN"])]
    Plot_arr = Lin_arr / Lin_arr.max()

    peaks = peaks[np.where(peaks == "LIN")[0]]


elif Stack_type == "PWS":

    PWS_arr, Results_arr, peaks = BF_XY_PWS(
        phase_traces=cut_phase_traces, degree=degree, **kwarg_dict
    )

    print(PWS_arr.max())

    peaks = np.c_[peaks, np.array(["PWS"])]

    Plot_arr = PWS_arr / PWS_arr.max()

    peaks = peaks[np.where(peaks == "PWS")[0]]


sx_min_plot = float(PRED_BAZ_X) + slow_min
sx_max_plot = float(PRED_BAZ_X) + slow_max
sy_min_plot = float(PRED_BAZ_Y) + slow_min
sy_max_plot = float(PRED_BAZ_Y) + slow_max


if Align == True:
    peaks[:, 0] = float(peaks[:, 0]) + float(PRED_BAZ_X)
    peaks[:, 1] = float(peaks[:, 1]) + float(PRED_BAZ_Y)

## write to file
filepath = Res_dir + "XY_Results.txt"

slow_vec_obs = []
for peak in peaks:

    slow_obs, baz_obs = get_slow_baz(float(peak[0]), float(peak[1]), dir_type="az")
    slow_vec_obs.append([baz_obs, slow_obs])

slow_vec_obs = np.array(slow_vec_obs).astype(float)
pred_file = np.array([BAZ, S]).astype(float)

write_to_file(
    filepath=filepath,
    st=st,
    peaks=slow_vec_obs,
    prediction=pred_file,
    phase=phase,
    time_window=window,
)

# plot!

with PdfPages(Res_dir + f"TP_Summary_Plot_{fmin:.2f}_{fmax:.2f}.pdf") as pdf:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    p = plotting(ax=ax)

    p.plot_TP_XY(
        tp=Plot_arr,
        peaks=peaks,
        sxmin=sx_min_plot,
        sxmax=sx_max_plot,
        symin=sy_min_plot,
        symax=sy_max_plot,
        sstep=s_space,
        contour_levels=20,
        title="%s Plot" % Stack_type,
        predictions=predictions,
        log=False,
    )
    pdf.savefig()
    plt.close()


    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    # filter stream
    st_record = st.copy()

    p = plotting(ax=ax2)
    p.plot_record_section_SAC(
        st=st_record, phase=phase, tmin=cut_min, tmax=cut_max, align=Align,
        type='distance'
    )

    ax2.vlines(
        x=t_min, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window min"
        , color='red'
    )

    ax2.vlines(
        x=t_max, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window max"
        , color='red'
    )

    ax2.legend(loc='best')

    pdf.savefig()
    plt.close()

    st_record = st.copy()

    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111)


    p = plotting(ax=ax3)
    p.plot_record_section_SAC(
        st=st_record, phase=phase, tmin=cut_min, tmax=cut_max, align=Align,
        type='az'
    )

    ax3.vlines(
        x=t_min, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], label="window min"
        , color='red'
    )

    ax3.vlines(
        x=t_max, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], label="window max"
        , color='red'
    )

    ax3.legend(loc='best')

    pdf.savefig()
    plt.close()


    ### plot great circle path

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection=ccrs.Robinson())

    # p = plotting(ax=ax)
    # p.plot_paths(st)

    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection=ccrs.Robinson())

    # p = plotting(ax=ax)
    # p.plot_stations(st)

    # pdf.savefig()
    # plt.close()
