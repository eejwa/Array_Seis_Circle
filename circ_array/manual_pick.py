
import numpy as np

import matplotlib.pyplot as plt

from array_info import array

# manually pick time window around phase
def pick_tw(stream, phase, tmin=-150, tmax=150, align=False):
    """
    Given an Obspy stream of traces, plot a record section and allow a time window to be picked around the phases of interest.

    Parameters
    ----------
    stream : Obspy stream object
        Stream of SAC files with travel time headers (tn)
        and labels (ktn) populated with gcarc and dist.

    phase : string
        Phase of interest (e.g. SKS)

    Returns
    -------
    window : 1D list
        The selected time window as numpy array [window_start, window_end].
    """

    # define a function to record the location of the clicks
    def get_window(event):
        """
        For an event such as a mouse click, return the x location of two events.

        event
            When creating interactive figure, an event will be a mouse click or key board press or something.

        Returns
        -------
        window : 1D list
            The selected time window as numpy array [window_start, window_end].
        """
        ix = event.xdata
        print("ix = %f" % ix)
        window.append(ix)
        # print(len(window))
        if np.array(window).shape[0] == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close()

        return window
    a = array(stream)
    # get the header with the times of the target phase in it
    Target_time_header = a.get_t_header_pred_time(phase=phase)

    # get the min and max predicted time of the phase at the array
    Target_phase_times, time_header_times = a.get_predicted_times(
        phase=phase
    )

    avg_target_time = np.mean(Target_phase_times)
    min_target_time = np.amin(Target_phase_times)
    max_target_time = np.amax(Target_phase_times)


    # plot a record section and pick time window
    # Window for plotting record section
    win_st = float(min_target_time + tmin)
    win_end = float(max_target_time + tmax)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    event_time = a.eventtime()
    stream_plot = stream.copy()
    stream_plot = stream_plot.trim(
        starttime=event_time + win_st, endtime=event_time + win_end
    )
    stream_plot = stream_plot.normalize()

    # plot each trace with distance
    for i, tr in enumerate(stream_plot):

        dist = tr.stats.sac.gcarc
        # if you want to align them, subtract the times of the target phase
        if align == True:
            tr_plot = tr.copy().trim(
                starttime=event_time
                + (getattr(tr.stats.sac, Target_time_header) + tmin),
                endtime=event_time + (getattr(tr.stats.sac, Target_time_header) + tmax),
            )
            time = np.linspace(tmin, tmax, int((tmax - tmin) * tr.stats.sampling_rate))
        else:
            tr_plot = tr.copy()
            time = np.linspace(
                win_st, win_end, int((win_end - win_st) * tr.stats.sampling_rate)
            )

        # reduce amplitude of traces and plot them
        dat_plot = tr_plot.data * 0.1
        # dat_plot = np.pad(
        #     dat_plot, (int(win_st * (1 / tr.stats.sampling_rate))), mode='constant')
        dat_plot += dist

        # make sure time array is the same length as the data
        if time.shape[0] != dat_plot.shape[0]:
            points_diff = -(abs(time.shape[0] - dat_plot.shape[0]))
            if time.shape[0] > dat_plot.shape[0]:
                time = np.array(time[:points_diff])
            if time.shape[0] < dat_plot.shape[0]:
                dat_plot = np.array(dat_plot[:points_diff])

        ax.plot(time, dat_plot, color="black", linewidth=0.5)

    # set x axis
    if align == True:
        plt.xlim(tmin, tmax)

    else:
        plt.xlim(win_st, win_end)

    #  plot the predictions
    for i, time_header in enumerate(time_header_times):
        t = np.array(time_header)

        if align == True:
            try:
                t[:, 0] = np.subtract(
                    t[:, 0].astype(float), np.array(Target_phase_times)
                )
            except:
                pass
        else:
            pass

        try:
            # sort array on distance
            t = t[t[:, 1].argsort()]
            ax.plot(
                t[:, 0].astype(float),
                t[:, 1].astype(float),
                color="C" + str(i),
                label=t[0, 2],
            )
        except:
            pass
            # print("t%s: No arrival" % i)

    # plt.title('Record Section Picking Window | Depth: %s Mag: %s' %(stream[0].stats.sac.evdp, stream[0].stats.sac.mag))
    plt.ylabel("Epicentral Distance ($^\circ$)")
    plt.xlabel("Time (s)")
    plt.legend(loc="best")

    window = []
    # turn on event picking package thing.
    cid = fig.canvas.mpl_connect("button_press_event", get_window)

    print("BEFORE YOU PICK!!")
    print("The first click of your mouse will the the start of the window")
    print("The second click will the the end of the window")
    plt.show()

    return window
