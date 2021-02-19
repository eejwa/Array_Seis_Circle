#!/usr/bin/env python

# clustering algorithms
from sklearn.cluster import dbscan

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

from array_plotting import plotting

import numpy as np

# import python packages
import obspy
from Parameters_Bootstrap import *

import circ_array as ca
from cluster_utilities import cluster_utilities
from circ_beam import shift_traces, linear_stack_baz_slow
from obspy.taup import TauPyModel

model = TauPyModel(model=pred_model)

st = obspy.read(filepath)

event_time = c.get_eventtime(st)
Target_phase_times, time_header_times = c.get_predicted_times(st, phase)

# the traces need to be trimmed to the same start and end time
# for the shifting and clipping traces to work (see later).
min_target = int(np.nanmin(Target_phase_times, axis=0)) + (-50)
max_target = int(np.nanmax(Target_phase_times, axis=0)) + (50)

stime = event_time + min_target
etime = event_time + max_target

# trim the stream
# Normalise and cut seismogram around defined window
st = st.copy().trim(starttime=stime, endtime=etime)
st = st.normalize()

evla = st[0].stats.sac.evla
evlo = st[0].stats.sac.evlo
evdp = st[0].stats.sac.evdp

distances = c.get_distances(st, type="deg")
mean_dist = np.mean(distances)
geometry = ca.get_geometry(st)
centre_lo, centre_la = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
sampling_rate = st[0].stats.sampling_rate

mean_lo = (evlo + centre_lo)/2
mean_la = (evla + centre_la)/2

# get predicted slownesses and backazimuths
predictions = ca.pred_baz_slow(stream=st, phases=phases, one_eighty=True)


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

st_traces = st.filter(type='bandpass', freqmin=fmin, freqmax=fmax,corners=1, zerophase=True)

Traces = c.get_traces(st_traces)

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


min_time = arrivals[0].time + t_min
arrival_times = cu.estimate_travel_times(traces=cut_shifted_traces,
                                 tmin=min_time,
                                 sampling_rate=sampling_rate,
                                 geometry=geometry,
                                 distance=mean_dist,
                                 pred_x=PRED_BAZ_X,
                                 pred_y=PRED_BAZ_Y)

rel_points = np.empty(All_Thresh_Peaks_arr.shape)
rel_points[:,0] = All_Thresh_Peaks_arr[:,0] - PRED_BAZ_X
rel_points[:,1] = All_Thresh_Peaks_arr[:,1] - PRED_BAZ_Y

cu_rel = cluster_utilities(labels=labels, points=rel_points)

means_xy, means_baz_slow = cu_rel.cluster_means()


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

    for times in arrival_times:
        print(np.mean(times))
        ax2.vlines(
            x=np.mean(times - arrivals[0].time), ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="mean_time_cluster"
            , color='blue'
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
        st=st_record, phase=phase, tmin=t_min, tmax=t_max, align=True,
        type='baz'
    )

    ax2.vlines(
        x=t_min, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window min"
        , color='red'
    )

    ax2.vlines(
        x=t_max, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="window max"
        , color='red'
    )

    for times in arrival_times:
        ax2.vlines(
            x=np.mean(times - arrivals[0].time), ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], label="mean_time_cluster"
            , color='blue'
        )

    ax2.set_title(f"Backazimuth Record Section Aligned on Predicted {phase}")


    ax2.legend(loc='best')
    pdf.savefig()
    plt.close()

    time_plot = np.arange(t_min, t_max, 1/sampling_rate)

    # plot lin_stacks of arrivals
    for i,arrival_mean in enumerate(means_baz_slow):
        baz = arrival_mean[0]
        slow = arrival_mean[1]
        print(baz)
        lin_stack = linear_stack_baz_slow(cut_shifted_traces, sampling_rate, geometry, mean_dist, slow, baz)

        fig = plt.figure(figsize=(10, 5))
        ax_stack = fig.add_subplot(111)
        ax_stack.plot(time_plot, lin_stack, color='black')
        ax_stack.set_title(f"Cluster {i}")
        ax_stack.set_xlabel("Time (s)")
        ax_stack.set_ylabel("Amplitude (m)")
        ax_stack.vlines(
            x=np.mean(arrival_times[i] - arrivals[0].time), ymin=ax_stack.get_ylim()[0], ymax=ax_stack.get_ylim()[1], label="mean_time_cluster"
            , color='blue'
        )

        ax_stack.vlines(
            x=np.mean(arrival_times[i] - arrivals[0].time) + (np.std(arrival_times[i]*2)), ymin=ax_stack.get_ylim()[0], ymax=ax_stack.get_ylim()[1], label="2 std dev"
            , color='blue', linewidth=0.5, linestyle='--'
        )

        ax_stack.vlines(
            x=np.mean(arrival_times[i] - arrivals[0].time) - (np.std(arrival_times[i]*2)), ymin=ax_stack.get_ylim()[0], ymax=ax_stack.get_ylim()[1], label="2 std dev"
            , color='blue', linewidth=0.5, linestyle='--'
        )


        pdf.savefig()
        plt.close()



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

    ### plot great circle path

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=mean_lo))

    p = plotting(ax=ax)
    p.plot_paths(st)

    pdf.savefig()
    plt.close()
