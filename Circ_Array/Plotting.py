
from Circ_Array import Circ_Array
c=Circ_Array()
import obspy
import matplotlib.pyplot as plt
import numpy as np
import scipy

class Plotting:
    def __init__(self):
        help="""
        This class will plot the record sections and outputs of the beamforming method.
        Feel free to add to this later.
        """

    def plot_record_section_SAC(self, st, phase, tmin=150, tmax=150, align=False):
        '''
        Plots a distance record section of all traces in the stream. The time window will
        be around the desired phase.

        Param: stream (Obspy Stream Object)
        Description: Stream of SAC files with the time (tn) and labels (ktn) populated.

        Param: phase (string)
        Description: The phase you are interested in analysing (e.g. SKS). Must be stored in the SAC headers tn and tkn.

        Param: tmin (float)
        Description: Time before the minumum predicted time of the phase you are interested in.

        Param: tmax (float)
        Description: Time after the maximum predicted time of the phase you are interested in.

        Plots record section, does not return anything.
        '''
        st = st.normalize()

        Target_time_header = c.get_t_header_pred_time(stream=st, phase=phase)

        Target_phase_times, time_header_times = c.get_predicted_times(
            stream=st, phase=phase)

        avg_target_time = np.mean(Target_phase_times)
        min_target_time = np.amin(Target_phase_times)
        max_target_time = np.amax(Target_phase_times)

        # plot a record section and pick time window
        # Window for plotting record section
        win_st = float(min_target_time - tmin)
        win_end = float(max_target_time + tmax)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        Y = st[0].stats.sac.nzyear
        JD = st[0].stats.sac.nzjday
        H = st[0].stats.sac.nzhour
        M = st[0].stats.sac.nzmin
        S = st[0].stats.sac.nzsec
        mS = st[0].stats.sac.nzmsec

        event_time = obspy.UTCDateTime(year=Y, julday=JD, hour=H,
                                 minute=M, second=S, microsecond=mS)
        stream_plot = st.copy
        stream_plot = st.trim(starttime=event_time + win_st,
                              endtime=event_time + win_end)

        for i, tr in enumerate(st):
            s_time = tr.stats.starttime
            e_time = tr.stats.endtime

            start = s_time - event_time
            end = e_time - event_time

            len = end - start

            dist = tr.stats.sac.gcarc
            if align == True:
                tr_plot = tr.copy().trim(starttime=event_time + (getattr(tr.stats.sac, Target_time_header) - tmin),
                              endtime=event_time + (getattr(tr.stats.sac, Target_time_header) + tmax))
                time = np.linspace(-tmin, tmax, int((tmin + tmax) * tr.stats.sampling_rate))
            else:
                tr_plot = tr.copy()
                time = np.linspace(win_st, win_end, int((win_end - win_st) * tr.stats.sampling_rate))

            dat_plot = tr_plot.data * 0.1
            dat_plot = np.pad(
                dat_plot, (int(start * (1 / tr.stats.sampling_rate))), mode='constant')
            dat_plot += dist

            # time = np.arange(0, end, 1/tr.stats.sampling_rate)
            if time.shape[0] != dat_plot.shape[0]:
                points_diff = -(abs(time.shape[0] - dat_plot.shape[0]))
                if time.shape[0] > dat_plot.shape[0]:
                    time = np.array(time[:points_diff])
                if time.shape[0] < dat_plot.shape[0]:
                    dat_plot = np.array(dat_plot[:points_diff])

            ax.plot(time, dat_plot, color='black', linewidth=0.5)

        if align == True:
            plt.xlim(-tmin, tmax)

        else:
            plt.xlim(win_st, win_end)

        for i,time_header in enumerate(time_header_times):
            t = np.array(time_header)

            if align == True:
                try:
                    t[:,0] = np.subtract(t[:,0].astype(float), np.array(Target_phase_times))
                    # sort array on distance
                except:
                    pass
            else:
                pass

            try:
                # sort array on distance
                t = t[t[:,1].argsort()]
                ax.plot(t[:, 0].astype(float),
                    t[:, 1].astype(float), color='C'+str(i), label=t[0, 2])
            except:
                print("t%s: No arrival" %i)


        # plt.title('Record Section Picking Window | Depth: %s Mag: %s' %(stream[0].stats.sac.evdp, stream[0].stats.sac.mag))
        plt.ylabel('Epicentral Distance ($^\circ$)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=14)
        plt.legend(loc='best')

        plt.show()

        return

    def add_circles(self, radii, x, y, colour, ax):
        """
        Adds circles of radius in the radii list with the centre at point xy with the defined colour
        to the axis the funtion is used in.

        Param: radii (array of floats)
        Description: array of circle radii.

        Param: x/y (float)
        Description: centre for circles.

        Param: colour (string)
        Description: colour of circles

        Returns nothing.
        """
        for r in radii:
            circle = plt.Circle((x, y), r, color=colour, clip_on=True,
                        fill=False, linestyle='--')
            ax.add_artist(circle)

        for b in range(45, 315, 60):
            ax.text((r) * np.sin(np.radians(b)), (r) *
                    np.cos(np.radians(b)), str(r), clip_on=True, color=colour, fontsize=10)

        return


    def add_lines(self, radius, x, y, angles, colour, ax):
        """
        Adds lines of length 'radius' from point xy with a variety of angles all
        with defined colour.

        Param: radius (float)
        Description: radius/length for lines.

        Param: x/y (float)
        Description: centre for lines.

        Param: angles (array floats)

        Param: colour (string)
        Description: colour of lines

        Returns nothing.
        """

        for a in angles:
            ax.plot([x, radius * np.cos(np.radians(a))],
                    [y, radius * np.sin(np.radians(a))], linestyle='--', color=colour, zorder=1)

        return



    def plot_TP_XY(self, tp, peaks, sxmin, sxmax, symin, symax, sstep, title, log = False, contour_levels=30, predictions=None):
        """
        Given a 2D array of power values for the theta-p analysis, it plots the
        values within the given slowness space with contours of power values.

        Param: tp (2D numpy array floats)
        Description: 2D array of power values of the theta-p plot.

        Param: peaks (1D numpy array floats)
        Description: x and y locations of the peak location.

        Param: s(x/y)min (float)
        Description: minimum x and y slowness value.

        Param: s(x/y)max (float)
        Description: maximum x and y slowness value.

        Param: sstep (float)
        Description: increment of steps in the slowness grid.

        Param: title (string)
        Description: title of the plot.

        Param: log (Bool)
        Description: True if you want the plot to be log scaled, False if linear scaling.

        Param: contour_levels (float)
        Description: number of contours.

        Param predictions (list of lists)
        Description: output of function 'pred_baz_slow' for the phases you
                     want to plot on the t-p plot.

        """

        steps_x = int(np.round((sxmax - sxmin) / sstep, decimals=0)) + 1
        steps_y = int(np.round((symax - symin) / sstep, decimals=0)) + 1
        slow_x = np.linspace(sxmin, sxmax, steps_x, endpoint=True)
        slow_y = np.linspace(symin, symax, steps_y, endpoint=True)

        if predictions is not None:
            Phases_x=predictions[:,3].astype(float)
            Phases_y=predictions[:,4].astype(float)
            Phases=predictions[:,0]

            Phases_x = np.where((Phases_x > sxmin) & (Phases_x < sxmax), Phases_x,Phases_x)
            Phases_y = np.where((Phases_y > symin) & (Phases_y < symax), Phases_y,Phases_y)
        else:
            pass


        radii = [2, 4, 6, 8, 10]
        angles = range(0, 360, 30)

        x_peaks = list(peaks[:,0].astype(float))
        y_peaks = list(peaks[:,1].astype(float))

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

        if log == True:
            ax.contourf(slow_x, slow_y, np.log(tp), contour_levels)
        elif log == False:
            ax.contourf(slow_x, slow_y, tp, contour_levels)
        else:
            pass

        ax.set_xlabel("p$_{x}$ (s/$^{\circ}$)", fontsize=14)
        ax.set_ylabel("p$_{y}$ (s/$^{\circ}$)", fontsize=14)
        ax.set_title(title, fontsize=14)
        self.add_circles(radii=radii, x=0, y=0, colour='w', ax=ax)
        self.add_lines(radius=10, x=0, y=0, angles=angles, colour='w', ax=ax)
        ax.scatter(x_peaks, y_peaks, color='red', marker='x', zorder=2)
        if predictions is not None:
            ax.scatter(Phases_x,Phases_y,color='white',s=20, zorder=3, marker='+')
            for i,p in enumerate(Phases):
                ax.text(Phases_x[i],Phases_y[i]-0.2,p , color='white', fontsize=10,zorder=3)
        else:
            pass

        ax.set_xlim(sxmin,sxmax)
        ax.set_ylim(symin,symax)
        plt.show()

        return



    def plot_TP_Pol(self, tp, peaks, smin, smax, bazmin, bazmax, sstep, bazstep, title, log = False, contour_levels=30, predictions=None):



        nslow = int(np.round(((smax - smin) / sstep) + 1))
        nbaz = int(np.round(((bazmax - bazmin) / bazstep) + 1))

        slows = np.linspace(smin, smax, nslow, endpoint=True)
        bazs = np.linspace(bazmin, bazmax, nbaz, endpoint=True)

        if predictions != None:
            Phases_b=predictions[:,2].astype(float)
            Phases_s=predictions[:,1].astype(float)
            Phases=predictions[:,0]

            Phases_b = np.where((Phases_b > bazmin) & (Phases_b < bazmax), Phases_b, Phases_b)
            Phases_s = np.where((Phases_s > smin) & (Phases_s < smax), Phases_s, Phases_s)
        else:
            pass

        b_peaks = list(peaks[:,0].astype(float))
        s_peaks = list(peaks[:,1].astype(float))


        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, polar=True)

        if log == True:
            ax.contourf(np.radians(bazs), slows, np.log(tp), contour_levels)
        elif log == False:
            ax.contourf(np.radians(bazs), slows, tp, contour_levels)
        else:
            pass

        ax.set_title(title, fontsize=16)

        ## Plot the radial axis label

        ax.scatter(np.radians(b_peaks), s_peaks, color='red', marker='x', zorder=2)

        if predictions != None:
            ax.scatter(np.radians(Phases_b),Phases_s,color='white',s=20, zorder=3, marker='+')
            for i,p in enumerate(Phases):
                ax.text(np.radians(Phases_b[i]),Phases_s[i]+0.2, p, color='white', fontsize=10,zorder=3)

        else:
            pass


        ax.set_thetalim(np.radians(bazmin),np.radians(bazmax))
        ax.set_rlim(smin,smax)
        ax.set_rorigin(0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.text(np.radians(bazmin - 7.5),(smin+smax)/2.,"$\it{p} \ (s/^{\circ})$", fontsize=14, rotation=90-bazmin, ha='center',va='center')
        ax.text(np.radians((bazmin+bazmax)/2),smin-0.25,"$\\theta \ (^{\circ})$", fontsize=14, rotation=180-((bazmax+bazmin)/2), ha='center',va='center')

        plt.show()


        return


    def plot_vespagram(self, vespagram, ymin, ymax, y_space, tmin, tmax, sampling_rate, title, type, predictions=None, npeaks=5, log = False, contour_levels=30, envelope=True):

        ny = int(np.round(((ymax - ymin) / y_space) + 1))
        ys = np.linspace(ymin, ymax, ny, endpoint=True)

        ntime = int(np.round(((tmax - tmin) * sampling_rate) + 1))
        times = np.linspace(tmin, tmax, ntime, endpoint=True)

        if predictions is not None:
            if type == 'slow':
                Phases_x=predictions[:,8].astype(float)
                Phases_y=predictions[:,1].astype(float)
                Phases=predictions[:,0]
            elif type == 'baz':
                Phases_x=predictions[:,8].astype(float)
                Phases_y=predictions[:,2].astype(float)
                Phases=predictions[:,0]
            else:
                print("type must be 'slow' or 'baz'")

            Phases_x = np.where((Phases_x > tmin) & (Phases_x < tmax), Phases_x,Phases_x)
            Phases_y = np.where((Phases_y > ymin) & (Phases_y < ymax), Phases_y,Phases_y)
        else:
            pass

        if envelope == True:
            for i,stack in enumerate(vespagram):
                vespagram[i] = obspy.signal.filter.envelope(stack)
        else:
            pass


        smoothed_vesp = scipy.ndimage.filters.gaussian_filter(
        vespagram, 2, mode='constant')
        peaks = c.findpeaks_XY(smoothed_vesp, xmin=tmin, xmax=tmax, ymin=ymin, ymax=ymax, xstep=(1/sampling_rate), ystep=y_space, N=npeaks)


        # Plot vespagram
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

        if log == True:
            v = ax.contourf(times, ys, np.log(vespagram), contour_levels)
        elif log == False:
            v = ax.contourf(times, ys, vespagram, contour_levels)
        else:
            pass

        plt.colorbar(v)

        if predictions is not None:
            ax.scatter(Phases_x,Phases_y,color='white',s=20, zorder=3, marker='+')
            for i,p in enumerate(Phases):
                ax.text(Phases_x[i],Phases_y[i]-0.2,p , color='white', fontsize=10,zorder=3)
        else:
            pass

        ax.scatter(peaks[:,0],peaks[:,1],marker='x',color='red',label="peaks")

        ax.set_title(title)
        ax.set_xlim(tmin,tmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(ymin,ymax)
        if type=='slow':
            ax.set_ylabel("p ($s/^{\circ}$)")
        elif type=='baz':
            ax.set_ylabel("$\\theta (^{\circ})$")
        else:
            print("type needs to be 'baz' or 'slow'")

        plt.show()
