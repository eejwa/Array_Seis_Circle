#!/usr/bin/env python

from circ_array import circ_array
from circ_beam import BF_Noise_Threshold_Relative_XY, shift_traces
import obspy
import matplotlib.pyplot as plt
import numpy as np
import scipy
from Parameters_Bootstrap import *
import random
import sys
import os

from obspy.taup import TauPyModel
model = TauPyModel(model=pred_model)
from mpi4py import MPI
c = circ_array()


## to be run with mpi
comm = MPI.COMM_WORLD

size_cores = comm.size
processes_per_core = int(Boots / size_cores)

# if directory does not exist, create it
if comm.Get_rank() == 0:
    if os.path.exists(Res_dir):
        pass
    else:
        os.makedirs(Res_dir)

st = obspy.read(filepath)
sample_size = len(st)

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
if Filt == True:
    st = st.filter('bandpass', freqmin=fmin, freqmax=fmax,
                      corners=4, zerophase=True)

else:
    pass

# get the traces and phase traces
Traces = c.get_traces(st)

# align the traces and phase traces
Shifted_Traces = shift_traces(traces=Traces, geometry=geometry, abs_slow=float(S), baz=float(BAZ),
                              distance=float(mean_dist), centre_x=float(centre_x), centre_y=float(centre_y),
                              sampling_rate=sampling_rate)

## cut the shifted traces within the defined time window
## from the t_min and rel_tmax

arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=mean_dist, phase_list=[phase])

pred_point = int(sampling_rate * (arrivals[0].time - min_target))
point_before= int(pred_point - (t_min * sampling_rate))
point_after= int(pred_point + (t_max * sampling_rate))

cut_shifted_traces = Shifted_Traces[:,point_before:point_after]

st_shifted = st.copy()
for index in range(len(st_shifted)):
    # replace unshifted data
    # keep information about geometry etc.
    st_shifted[index].data = cut_shifted_traces[index]



# make lists to store the peaks, noise and theta-p plots

Lin_list = []
Noise_list = []
threshold_peaks_list = []


for i in range(0, processes_per_core):
    print(i)


    label = (i * int(comm.size)) + int(comm.rank)
    # get a random sample of traces - needs to be in list format
    st_rank_list = random.choices(list(st_shifted), k=sample_size)

    # change it back into an obspy Stream
    st_rank = obspy.Stream(traces=st_rank_list)
    st_rank = st_rank.normalize()
    ntrace = len(st_rank)

    Traces = []
    geometry = []
    distances = []


    # pre-process and record station information
    for i, tr in enumerate(st_rank):

        tr.data = tr.data.astype(float)

        ep_dist = tr.stats.sac.gcarc
        stla = tr.stats.sac.stla
        stlo = tr.stats.sac.stlo
        stel = tr.stats.sac.stel
        geometry.append([stlo, stla, stel])
        distances.append(ep_dist)

        Traces.append(tr.data)

    Traces = np.array(Traces)
    geometry = np.array(geometry)
    distances = np.array(distances)
    avg_dist = np.mean(distances)

    # get centre
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
    geometry[:, 1]), np.mean(geometry[:, 2])

    lin_tp, noise_mean, max_peak =  BF_Noise_Threshold_Relative_XY(traces=Traces,
                                                                   sampling_rate=sampling_rate,
                                                                   geometry=geometry, distance=mean_dist,
                                                                   sxmin=slow_min, sxmax=slow_max,
                                                                   symin=slow_min, symax=slow_max,
                                                                   s_space=s_space)

    Threshold_lin_array = np.copy(lin_tp)
    Threshold_lin_array[lin_tp <=noise_mean * 3] = 0

    Smoothed_thresh_lin_array = scipy.ndimage.filters.gaussian_filter(
                                Threshold_lin_array, 1, mode='constant')


    sx_min = float(PRED_BAZ_X) + slow_min
    sx_max = float(PRED_BAZ_X) + slow_max
    sy_min = float(PRED_BAZ_Y) + slow_min
    sy_max = float(PRED_BAZ_Y) + slow_max

    peaks = c.findpeaks_XY(Smoothed_thresh_lin_array, xmin=sx_min,
                           xmax=sx_max, ymin=sy_min, ymax=sy_max,
                           xstep=s_space, ystep=s_space, N=peak_number)


    no_smooth_peaks = c.findpeaks_XY(Threshold_lin_array, xmin=sx_min,
                           xmax=sx_max, ymin=sy_min, ymax=sy_max,
                           xstep=s_space, ystep=s_space, N=peak_number)

    sys.stdout.flush()

    # add the arrays to a list
    Lin_list.append(lin_tp)
    Noise_list.append(noise_mean)
    threshold_peaks_list.append(peaks)

Lin_mean = np.mean(np.array(Lin_list), axis=0)

threshold_peaks_arr = np.array(threshold_peaks_list)
threshold_peaks_arr = np.vstack(threshold_peaks_arr)

comm.Barrier()   # wait for all cores to synchronize here


all_lin_list_combined = np.array(comm.gather(Lin_mean, root=0))
all_noise_list_combined = np.array(comm.gather(Noise_list, root=0))
all_thresh_peaks_list_combined = np.array(
    comm.gather(threshold_peaks_arr, root=0))


if comm.Get_rank() == 0:
    Lin_Mean = np.mean(all_lin_list_combined, axis=0)
    # print(Lin_Mean.shape)
    # Lin_list = []
    Noise_list = []
    Peaks_list = []

    All_Thresh_Peaks_arr = np.vstack(all_thresh_peaks_list_combined)
    All_Lin_arr = np.array(Lin_list)
    All_Noise_arr = np.array(Noise_list)

    # Ok need to get all the probability arrays into one list/array.
    all_prob_list = []
    for x in range(0, comm.size):
        for y in range(0, processes_per_core):
            Noise_list.append(all_noise_list_combined[x][y])


    # save to the results directory!
    all_max_peaks_filename = "%sAll_Thresh_Peaks_%s_%s_%s_array" %(Res_dir, Boots, fmin, fmax)
    all_lin_arr_filename = "%sAll_Lin_%s_%s_%s_array" %(Res_dir, Boots, fmin, fmax)
    all_noise_arr_filename = "%sAll_Noise_%s_%s_%s_array" %(Res_dir, Boots, fmin, fmax)

    n_array_names = [all_max_peaks_filename, all_lin_arr_filename, all_noise_arr_filename]

    for a_name in n_array_names:
        if os.path.exists(a_name):
                os.remove(a_name)

    np.save(all_max_peaks_filename, All_Thresh_Peaks_arr)
    np.save(all_lin_arr_filename, All_Lin_arr)
    np.save(all_noise_arr_filename, All_Noise_arr)
