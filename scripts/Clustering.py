#!/usr/bin/env python

# clustering algorithms
from sklearn.cluster import dbscan

import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

from array_plotting import plotting

import numpy as np

# import python packages
import obspy
from Parameters_Bootstrap import *

from array_info import array
from cluster_utilities import cluster_utilities
from shift_stack import shift_traces, linear_stack_baz_slow
from obspy.taup import TauPyModel

from slow_vec_calcs import get_slow_baz

from vespagram import Vespagram_Lin


model = TauPyModel(model=pred_model)

st = obspy.read(filepath)
a = array(st)
event_time = a.eventtime()
Target_phase_times, time_header_times = a.get_predicted_times(phase)

# the traces need to be trimmed to the same start and end time
# for the shifting and clipping traces to work (see later).
min_target = int(np.nanmin(Target_phase_times, axis=0)) + (-100)
max_target = int(np.nanmax(Target_phase_times, axis=0)) + (100)

stime = event_time + min_target
etime = event_time + max_target

# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()

evla = st[0].stats.sac.evla
evlo = st[0].stats.sac.evlo
evdp = st[0].stats.sac.evdp

distances = a.distances(type="deg")
mean_dist = np.mean(distances)
geometry = a.geometry()
centre_lo, centre_la = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
sampling_rate = st[0].stats.sampling_rate

mean_lo = (evlo + centre_lo)/2
mean_la = (evla + centre_la)/2

# get predicted slownesses and backazimuths
predictions = a.pred_baz_slow(phases=phases, one_eighty=True)


# find the line with the predictions for the phase of interest
row = np.where((predictions == phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[
    row, :
][0]
PRED_BAZ_X = float(PRED_BAZ_X)
PRED_BAZ_Y = float(PRED_BAZ_Y)
S = float(S)
BAZ = float(BAZ)


All_Noise_arr = np.load(
    numpy_dir + "All_Noise_%s_%s_%s_array.npy" % (Boots, fmin, fmax)
)
All_Thresh_Peaks_arr = np.load(
    numpy_dir + "All_Thresh_Peaks_%s_%s_%s_array.npy" % (Boots, fmin, fmax)
)
Lin_Mean = np.load(numpy_dir + "Mean_Lin_%s_%s_%s_array.npy" % (Boots, fmin, fmax))

core_samples, labels = dbscan(
    X=All_Thresh_Peaks_arr, eps=epsilon, min_samples=int(Boots * MinPts)
)
no_clusters = np.amax(labels) + 1

cu = cluster_utilities(labels=labels, points=All_Thresh_Peaks_arr)

new_labels = cu.remove_noisy_arrivals(st=st, phase=phase, slow_vec_error=slow_vec_error)

## recover travel time estimates

# get traces

st_traces = st.filter(type='bandpass', freqmin=fmin, freqmax=fmax, corners=1, zerophase=True)
array_new = array(st_traces)
Traces = array_new.traces()

# align the traces
Shifted_Traces = shift_traces(
    traces=Traces,
    geometry=geometry,
    abs_slow=float(S),
    baz=float(BAZ),
    distance=float(mean_dist),
    centre_x=float(centre_lo),
    centre_y=float(centre_la),
    sampling_rate=sampling_rate,
    elevation=True,
    incidence=8
)


## cut the shifted traces within the defined time window
## from the t_min and rel_tmax

arrivals = model.get_travel_times(
    source_depth_in_km=evdp,
    distance_in_degree=mean_dist,
    phase_list=[phase]
)

pred_point = int(sampling_rate * (arrivals[0].time - min_target))
point_before = int(pred_point + (t_min * sampling_rate))
point_after = int(pred_point + (t_max * sampling_rate))

cut_shifted_traces = Shifted_Traces[:, point_before:point_after]

for shtr in cut_shifted_traces:
    shtr /= shtr.max()


min_time = arrivals[0].time + t_min
cu_times = cluster_utilities(labels=new_labels, points=All_Thresh_Peaks_arr)
arrival_times = cu_times.estimate_travel_times(traces=cut_shifted_traces,
                                 tmin=min_time,
                                 sampling_rate=sampling_rate,
                                 geometry=geometry,
                                 distance=mean_dist,
                                 pred_x=PRED_BAZ_X,
                                 pred_y=PRED_BAZ_Y)
rel_points = np.empty(All_Thresh_Peaks_arr.shape)
rel_points[:,0] = All_Thresh_Peaks_arr[:,0] - PRED_BAZ_X
rel_points[:,1] = All_Thresh_Peaks_arr[:,1] - PRED_BAZ_Y

cu_rel = cluster_utilities(labels=new_labels, points=rel_points)


means_xy, means_baz_slow = cu_rel.cluster_means()
mean_baz = means_baz_slow[0][0]

# estimate times without aligning tracesfor sanity
# times = cu.estimate_travel_times(traces=Traces,
#                                  tmin=min_target,
#                                  sampling_rate=sampling_rate,
#                                  geometry=geometry,
#                                  distance=mean_dist,
#                                  pred_x=0,
#                                  pred_y=0)

x_min = PRED_BAZ_X + slow_min
x_max = PRED_BAZ_X + slow_max
y_min = PRED_BAZ_Y + slow_min
y_max = PRED_BAZ_Y + slow_max

# write to results file

Results_filepath = Res_dir + f"Clustering_Results_{fmin:.2f}_{fmax:.2f}.txt"

Results_filepath_unfiltered = Res_dir + f"Clustering_Results_{fmin:.2f}_{fmax:.2f}_unfiltered.txt"

# make new lines list

newlines = cu.create_newlines(
    st=st,
    file_path=Results_filepath,
    phase=phase,
    window=[t_min, t_max],
    Boots=Boots,
    epsilon=epsilon,
    slow_vec_error=slow_vec_error,
    Filter=True,
)

newlines_unfiltered = cu.create_newlines(
    st=st,
    file_path=Results_filepath_unfiltered,
    phase=phase,
    window=[t_min, t_max],
    Boots=Boots,
    epsilon=epsilon,
    slow_vec_error=slow_vec_error,
    Filter=False,
)

# cluster times 

rel_times = arrival_times - arrivals[0].time
core_samples_time, labels_time = dbscan(
    X=rel_times[0].reshape(-1, 1), eps=1, 
    min_samples=int(Boots * MinPts)
)

print(arrivals[0].time)
print(arrival_times)
print(rel_times)

slowness_cluster = cu.group_points_clusters()[0]
print(slowness_cluster[:,1])


slows = np.sqrt((slowness_cluster[:,0] ** 2) + (slowness_cluster[:,1] ** 2))
azimuths = np.degrees(np.arctan2(slowness_cluster[:,0], slowness_cluster[:,1]))  # * (180. / math.pi)

# % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
bazs = azimuths % -360 + 180

print(mean_baz)

vesp_lin = Vespagram_Lin(Traces,  sampling_rate = sampling_rate, geometry=geometry, 
                         distance=mean_dist, baz=float(mean_baz), smin=0, smax=5, s_space=0.05)

ny = int(np.round(((0 + 5) / 0.05) + 1))
ys = np.linspace(0, 5, ny, endpoint=True)
ntime = int(np.round(((t_max - t_min) * sampling_rate)))
vesp_times = np.linspace(t_min, t_max, ntime, endpoint=True) + arrivals[0].time
plot_times = np.arange(min_target, max_target + (1/sampling_rate), 1/sampling_rate)

for i, stack in enumerate(vesp_lin):
    vesp_lin[i] = obspy.signal.filter.envelope(stack)

# df = pd.DataFrame({"Slow_x":All_Thresh_Peaks_arr[:,0], "Slow_y":All_Thresh_Peaks_arr[:,1], "Labels":new_labels})

# plot!
with PdfPages(Res_dir + f"Clustering_Summary_Plot_{fmin:.2f}_{fmax:.2f}.pdf") as pdf:
    # clusters
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    p = plotting(ax=ax1)

    p.plot_clusters_XY(
        labels=new_labels,
        tp=Lin_Mean,
        peaks=All_Thresh_Peaks_arr,
        sxmin=x_min,
        sxmax=x_max,
        symin=y_min,
        symax=y_max,
        sstep=s_space,
        title="Clusters",
        predictions=predictions,
        ellipse=True,
        std_devs=[1, 2, 3],
    )

    pdf.savefig()
    plt.close()


    # fig = plt.figure(figsize=(10, 10))
    # ax1 = fig.add_subplot(111)
    # sns.jointplot(
    # data=df,
    # x="Slow_x", y="Slow_y",
    # kind="scatter"
    # )
    # pdf.savefig()
    # plt.close()

    # record section
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    # filter stream
    st_record = st.filter(type='bandpass', freqmin=fmin, freqmax=fmax,corners=1, zerophase=True)
    p = plotting(ax=ax2)
    p.plot_record_section_SAC(
        st=st_record, phase=phase, tmin=-50, tmax=50, align=True,
        type='distance', scale=0.01
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

    ax2.hlines(
        y=mean_dist, xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], label="mean dist"
        , color='green'
    )

    ax2.set_title(f"Distance Record Section Aligned on Predicted {phase}")

    pdf.savefig()
    plt.close()



    # backazimuth record section
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    p = plotting(ax=ax2)
    p.plot_record_section_SAC(
        st=st_record, phase=phase, tmin=-50, tmax=50, align=True,
        type='baz', scale=0.01
    )

    ax2.vlines(
        x=t_min, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window min"
        , color='red'
    )

    ax2.vlines(
        x=t_max, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window max"
        , color='red'
    )

    ax2.set_title(f"Backazimuth Record Section Aligned on Predicted {phase}")


    ax2.legend(loc='best')
    pdf.savefig()
    plt.close()

    time_plot = np.arange(t_min, t_max, 1/sampling_rate)

    ### plot stations

    fig = plt.figure(figsize=(6, 6))

    # need to give a projection for the cartopy package
    # to work. See a list here:
    # https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=mean_lo))

    p = plotting(ax=ax)
    p.plot_stations(st)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    pdf.savefig()
    plt.close()

    ## plot great circle path

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=mean_lo))

    p = plotting(ax=ax)
    p.plot_paths(st)

    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)
    ax2.hist(rel_times[0], 30, color='blue')
    ax2.set_xlabel("Relative arrival times (s)", fontsize=14)
    ax2.set_ylabel("Counts", fontsize=14)
    plt.savefig("travel_time_histogram.pdf")
    pdf.savefig()
    plt.close()

    # plot the clusters found from their arrival times
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    # v = ax2.contourf(plot_times[point_before:point_after], ys, vesp_lin[:, point_before:point_after], 50)
    for p in range(np.amax(labels_time) + 1):
        # if p == 0:
        #     ax2.scatter(rel_times[0][np.where(labels_time == p)], slows[np.where(labels_time == p)], color='black', label=)
        # else: 
        ax2.scatter(rel_times[0][np.where(labels_time == p)],  slows[np.where(labels_time == p)], label=f"Cluster {p}")
    
    ax2.set_xlim([t_min, t_max])
    # ax2.set_ylim([2, 5])
    ax2.set_xlabel(f"Window Time Relative to {phase} (s)", fontsize=14)
    ax2.set_ylabel("$p$ ($s/^{\circ}$)", fontsize=14)
    plt.legend(loc='best')
    plt.savefig("hslow_time_distribution.pdf")
    pdf.savefig()
    plt.close()

        # plot the clusters found from their arrival times
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111)

    # v = ax2.contourf(plot_times[point_before:point_after], ys, vesp_lin[:, point_before:point_after], 50)
    for p in range(np.amax(labels_time) + 1):
        # if p == 0:
        #     ax2.scatter(rel_times[0][np.where(labels_time == p)], bazs[np.where(labels_time == p)], color='black', label='noise')
        # else: 
        ax2.scatter(rel_times[0][np.where(labels_time == p)],  bazs[np.where(labels_time == p)], label=f"Cluster {p}")
    
    ax2.set_xlim([t_min, t_max])
    # ax2.set_ylim([1, 5])
    ax2.set_xlabel(f"Window Time Relative to {phase} (s)", fontsize=14)
    ax2.set_ylabel("$\\theta$ ($^{\circ}$)", fontsize=14)
    plt.legend(loc='best')
    plt.savefig("baz_time_distribution.pdf")
    pdf.savefig()
    plt.close()

