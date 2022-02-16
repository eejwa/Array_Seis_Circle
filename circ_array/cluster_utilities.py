import numpy as np
from scipy.stats import circmean, circstd, circvar
import obspy
from slow_vec_calcs import get_slow_baz
from array_info import array
from utilities import myround
from geo_sphere_calcs import relocate_event_baz_slow
from shift_stack import linear_stack_baz_slow, shift_traces
import obspy.signal.filter as o
from sklearn.neighbors import KDTree
import os
from obspy.taup import TauPyModel





class cluster_utilities:

    description = """
    This class holds functions to give information about the clusters such as their mean,
    standard deviations, area, ellipse properties.

    Attributes
    ----------
    labels : 1D array of ints
        Contains the cluster labels for each datapoint
        e.g. labels = np.array([0, 0, 1, 1, 1, 0])

    points : 2D numpy array
        Contains datapoints used for the cluster analysis
        e.g. points = np.array([[0,0], [1,0], [1,2], [2,2], [3,2], [0,1]])
    """

    def __init__(self, labels, points):
        """
        All functions require the data points and their associated labels.
        Labels are given as output of clustering algorithms and are a list
        of integers. Each integer represents a cluster.

        """
        # assert(labels.shape == points.shape[0])
        self.labels = labels
        self.points = points

    def group_points_clusters(self):
        """
        From the points and labels, this function will split the points
        into n arrays (n being the number of clusters) and store them
        in an array.

        Parameters
        ----------

        None

        Returns
        -------
            points_cluters : 3D array of floats
                Each row has a 2D array of points for a
                particular cluster.
        """

        points_clusters = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )

            points_cluster = np.array([points_x, points_y]).T
            points_clusters.append(points_cluster)

        return points_clusters

    def eigsorted(self, cov):
        """
        Given a covariance matrix, calculate the eigenvalues and eigenvectors.


        Parameters
        ----------
        cov : 2D numpy array of floats
            Covariance matrix of an ndarray of points.
            Can be found from the np.cov function.

        Returns
        -------
            vals : 2D array of floats
                Eigenvalues in decending order.
            vecs : 2D array of floats
                Eigenvectors in decending order.
        """

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]  # create decending order
        # return eigenvalues and eigenvectors in decending order
        return vals[order], vecs[:, order]

    def cluster_means(self):
        """
        Given data points and cluster labels, return the mean slowness vector
        properties of each cluster.

        Parameters
        ----------
        points : 2D numpy array of floats.
            Array of points used to find clusters.

        labels : 1D numpy array of integers.
            Labels of each point for which cluster it is in.

        rel_x : float
            x component of slowness vector used to align traces for relative beamforming.

        rel_y : float
            y component of slowness vector used to align traces for relative beamforming.

        Returns
        -------
                means_xy : 2D array of floats
                    Mean slow_x and slow_y for each cluster.

                means_baz_slow : 2D array of floats
                    Mean backazimuth and horizontal slowness
                    for each cluster.
        """

        means_xy = []
        means_baz_slow = []

        for p in range(np.amax(self.labels) + 1):

            p_x, p_y = (
                np.mean(self.points[np.where(self.labels == p)][:, 0]),
                np.mean(self.points[np.where(self.labels == p)][:, 1]),
            )

            mean_abs_slow, mean_baz = get_slow_baz(p_x, p_y, dir_type="az")

            mean_abs_slow = np.around(mean_abs_slow, 2)
            mean_baz = np.around(mean_baz, 1)

            means_xy.append(np.array([p_x, p_y]))
            means_baz_slow.append(np.array([mean_baz, mean_abs_slow]))

        return np.around(means_xy, 2), np.around(means_baz_slow, 2)

    def cluster_std_devs(self, pred_x, pred_y):
        """
        Given data points and cluster labels, return the standard deviations of
        backazimuth and horizontal slownesses of each cluster.

        Parameters
        ----------
            pred_x : float
                   x component of the predicted slowness vector.

            pred_y : float
                   y component of the predicted slowness vector.


        Returns
        -------
                bazs_std : 1D array of floats
                    The standard deviation for backazimuth values in each cluster.

                slows_std : 1D array of floats
                    The standard deviation for horizontal slowness values
                    in each cluster.
        """

        bazs_std = []
        slows_std = []
        slow_x_std = []
        slow_y_std = []
        azs_std = []
        mags_std = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )

            rel_xs = points_x - pred_x
            rel_ys = points_y - pred_y

            azs = np.degrees(np.arctan2(rel_ys,rel_xs))
            mags = np.sqrt(rel_xs**2 + rel_ys**2)

            azs = np.where(azs<0, azs + 360, azs)

            az_std = np.around(np.degrees(circstd(np.radians(azs))), 1)
            mag_std = np.std(mags)

            x_std = np.std([points_x])
            y_std = np.std([points_y])

            points_cluster = np.array([points_x, points_y]).T

            abs_slows = np.zeros(points_x.shape)
            bazs = np.zeros(points_x.shape)

            for i,point_x in enumerate(points_x):

                point_y = points_y[i]

                abs_slow, baz = get_slow_baz(point_x, point_y, dir_type="az")
                abs_slows[i] = abs_slow
                bazs[i] = baz

            abs_slow_std_dev = np.around(np.std(abs_slows), 2)
            # stdev - use scipy's circstd
            baz_std_dev = np.around(np.degrees(circstd(np.radians(bazs))), 1)

            bazs_std.append(baz_std_dev)
            slows_std.append(abs_slow_std_dev)
            slow_x_std.append(x_std)
            slow_y_std.append(y_std)
            azs_std.append(az_std)
            mags_std.append(mag_std)

        return np.around(bazs_std, 2), np.around(slows_std, 2), np.around(slow_x_std, 2), np.around(slow_y_std, 2), np.around(azs_std, 2), np.around(mags_std, 2)

    def covariance_matrices(self):
        """
        Given the labels and point locations, returns the covariance
        matrices for each cluster.

        Parameters
        ----------
        rel_x : float
               x component of slowness vector used to align traces for relative beamforming.

        rel_y : float
               y component of slowness vector used to align traces for relative beamforming.

        Returns
        -------
            covariance_matrices : 1D array of floats
                Covariance matrices.
        """

        covariance_matrices = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )
            #
            # points_x += rel_x
            # points_y += rel_y

            co_mat = np.cov(np.array([points_x, points_y]).T, rowvar=False)

            covariance_matrices.append(co_mat)

        covariance_matrices = np.array(covariance_matrices)

        return covariance_matrices

    def cluster_ellipse_properties(self, std_dev):
        """
        Given data points and cluster labels, return the area of the error ellipse with
        the given standard deviation.

        Parameters
        ----------
        std_dev : int
                 Standard deviation of error ellipse (typically 1,2, or 3).

        Returns
        -------
                ellipse_properties 2D array of the error ellipse widths heights and thetas.
                                     [[i, width, height, theta], ..] for i in number_of_clusters.

        """

        ellipse_properties = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            co_mat = np.cov(np.array([points_x, points_y]).T, rowvar=False)

            # calculate the area of the ellipse
            vals, vecs = self.eigsorted(co_mat)

            # calculate the width and height of the ellipse
            width, height = 2 * std_dev * np.sqrt(vals)

            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            ellipse_properties.append([p, width, height, theta])

        return np.around(ellipse_properties, 2)

    def cluster_ellipse_areas(self, std_dev):
        """
        Given data points and cluster labels, return the area of the error ellipse with
        the given standard deviation.

        Parameters
        ----------
        std_dev : int.
                 Standard deviation of error ellipse (typically 1,2, or 3).

        rel_x : float
               x component of slowness vector used to align traces for relative beamforming.

        rel_y : float
               y component of slowness vector used to align traces for relative beamforming.

        Returns
        -------
                ellipse_areas 1D array of the area for the error ellipse of the points in each cluster.

        """

        ellipse_areas = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            co_mat = np.cov(np.array([points_x, points_y]).T, rowvar=False)

            # calculate the area of the ellipse
            vals, vecs = self.eigsorted(co_mat)

            # calculate the width and height of the ellipse
            width, height = 2 * std_dev * np.sqrt(vals)

            Area_error_ellipse = np.pi * (width / 2) * (height / 2)

            ellipse_areas.append(Area_error_ellipse)

        return np.around(ellipse_areas, 2)

    def cluster_baz_slow_95_conf(self, std_dev):
        """
        Given data points and cluster labels, return the 95% confidence range for baz and
        horizontal slowness in the cluster.

        Parameters
        ----------
        points : 2D numpy array of floats.
                Array of points used to find clusters.

        labels : 1D numpy array of integers.
                Labels of each point for which cluster it is in.

        std_dev : int
                 Standard deviation of error ellipse (typically 1,2, or 3).

        rel_x : float
               x component of slowness vector used to align traces for relative beamforming.

        rel_y : float
               y component of slowness vector used to align traces for relative beamforming.

        Returns
        -------
                bazs_95_confidence 2D array of floats. Each row contains the upper and lower
                                     bounds for the 95% backazimuth confidence intervals.
                                     Each row represents a clusters.

                slows_95_confidence 2D array of floats. Each row contains the upper and lower
                                     bounds for the 95% horizontal slowness confidence intervals.
                                     Each row represents a clusters.

        """

        bazs_95_confidence = []
        slows_95_confidence = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            abs_slows, bazs = get_slow_baz_array(points_x, points_y, dir_type="az")

            mean_abs_slow = np.around(mean_abs_slow, 2)
            mean_baz = np.around(mean_baz, 1)

            abs_slows_sorted = np.sort(abs_slows)
            lower_abs_slow = np.around(np.percentile(abs_slows_sorted, 2.5), 2)
            upper_abs_slow = np.around(np.percentile(abs_slows_sorted, 97.5), 2)

            baz_diff = 360.0 - mean_baz

            # now rotate all angles by this so the mean baz is at 0/360
            rotated_bazs = (bazs + baz_diff) % 360

            # Now, using arctan2(), can find the angle difference frmo -pi to +pi
            diffs = np.degrees(
                np.arctan2(
                    np.sin(np.radians(rotated_bazs)), np.cos(np.radians(rotated_bazs))
                )
            )

            # order the diffs and get the percentiles
            diffs_sorted = np.sort(diffs)
            diffs_lower = np.percentile(diffs_sorted, 2.5)
            diffs_upper = np.percentile(diffs_sorted, 97.5)

            # convert this to values relative to baz mean (?)
            baz_lower = np.around(mean_baz + diffs_lower, 1)
            baz_upper = np.around(mean_baz + diffs_upper, 1)

            bazs_95_confidence.append([baz_lower, baz_upper])
            slows_95_confidence.append([lower_abs_slow, upper_abs_slow])

        return np.array(bazs_95_confidence), np.array(slows_95_confidence)


    def remove_noisy_arrivals(self, st, phase, slow_vec_error=3):
        """
        Removes arrivals/cluter based on their slowness vector deviations.

        Parameters
        ----------
        st : Obspy stream object
             Obspy stream object of sac files with event, arrival time and
             station headers populated.

        phase : string
                Target phase (e.g. SKS)

        slow_vec_error : float
                         Maximum slowness vector deviation between the predicted
                         and observed arrival. Arrival with larger deviations will
                         be removed if Filter = True (below). Default is 3.

        Returns
        -------
        labels : 1D numpy array of floats
            1D numpy array of the labels describing which points are
            noise and which are clusters.
        """
        from array_info import array
        array = array(st)
        # get predicted slownesses and backazimuths
        predictions = array.pred_baz_slow(phases=[phase], one_eighty=True)

        # find the line with the predictions for the phase of interest
        row = np.where((predictions == phase))[0]

        (
            P,
            S,
            BAZ,
            PRED_BAZ_X,
            PRED_BAZ_Y,
            PRED_AZ_X,
            PRED_AZ_Y,
            DIST,
            TIME,
        ) = predictions[row, :][0]
        PRED_BAZ_X = float(PRED_BAZ_X)
        PRED_BAZ_Y = float(PRED_BAZ_Y)


        no_clusters = np.amax(self.labels) + 1
        means_xy, means_baz_slow = self.cluster_means()
        updated_labels = self.labels
        if no_clusters != 0:
            for i in range(no_clusters):

                slow_x_obs = myround(means_xy[i, 0])
                slow_y_obs = myround(means_xy[i, 1])

                del_x_slow = slow_x_obs - PRED_BAZ_X
                del_y_slow = slow_y_obs - PRED_BAZ_Y

                distance = np.sqrt(del_x_slow ** 2 + del_y_slow ** 2)

                if distance > slow_vec_error:
                    ## change the labels
                    updated_labels = np.where(updated_labels == i, -1, updated_labels)
                else:
                    pass
        return updated_labels

    def estimate_travel_times(self,
                              traces,
                              tmin,
                              sampling_rate,
                              geometry,
                              distance,
                              pred_x,
                              pred_y):
        """
        Given cluster information, estimate the arrival times of each
        point in the cluster. Note this assumes the traces have been shifted relative
        to a predicted arrival and the cluster points are relative to this
        predicted arrival.



        """

        times_arrivals = []

        # for each cluster
        for p in range(np.amax(self.labels) + 1):
            times = []
            points_x, points_y = (
                self.points[np.where(self.labels == p)][:, 0],
                self.points[np.where(self.labels == p)][:, 1],
            )


            points_x -= pred_x
            points_y -= pred_y

            abs_slows = np.zeros(points_x.shape)
            bazs = np.zeros(points_x.shape)

            for i,point_x in enumerate(points_x):

                point_y = points_y[i]

                abs_slow, baz = get_slow_baz(point_x, point_y, dir_type="az")
                abs_slows[i] = abs_slow
                bazs[i] = baz


            for i,baz in enumerate(bazs):
                slow = abs_slows[i]

                # shift and stack the traces linearly along the
                # backazimuth and slowness

                lin_stack = linear_stack_baz_slow(traces, sampling_rate, geometry, distance, slow, baz)

                # calculate the envelope of this stack and recover the maximum

                data_envelope = abs(lin_stack)

                # get the index of the max value
                imax = np.where(data_envelope == np.amax(data_envelope))[0][0]

                # get this index in seconds and add to the start of the window
                time = (imax / sampling_rate) + tmin

                times.append(time)
                # import matplotlib.pyplot as plt
                # plt.plot(lin_stack)
                # plt.title(imax)
                # plt.show()

            times = np.array(times)
            times_arrivals.append(times)
        return times_arrivals

    def create_newlines(
        self,
        st,
        file_path,
        phase,
        window,
        Boots,
        epsilon,
        slow_vec_error=3,
        Filter=False,
    ):
        """
        This function will create a list of lines with all relevant information to be stored in
        the results file. The function write_to_cluster_file() will write these lines to a new
        file and replace any of the lines with the same array location and target phase.

        Parameters
        ----------
        st : Obspy stream object
             Obspy stream object of sac files with event, arrival time and
             station headers populated.

        file_path : string
                    Path to the results file to check the contents of.

        phase : string
                Target phase (e.g. SKS)

        window : list of floats
                 tmin and tmax describing the relative time window.

        Boots : int
                Number of bootstrap samples.
        epsilon : float
                  Epsilon value used to find the clusters.

        slow_vec_error : float
                         Maximum slowness vector deviation between the predicted
                         and observed arrival. Arrival with larger deviations will
                         be removed if Filter = True (below). Default is 3.

        Filter : bool
                 Do you want to filter out the arrivals (default = False)

        Returns
        -------
            newlines: list of strings of the contents to write to the results file.

        """

        model = TauPyModel(model='prem')

        newlines = []
        header = ("Name evla evlo evdp reloc_evla reloc_evlo "
                  "stla_mean stlo_mean slow_pred slow_max slow_diff "
                  "slow_std_dev baz_pred baz_max baz_diff baz_std_dev "
                  "slow_x_pred slow_x_obs del_x_slow x_std_dev slow_y_pred slow_y_obs "
                  "del_y_slow y_std_dev az az_std mag mag_std time_obs time_pred time_diff time_std_dev "
                  "error_ellipse_area ellispe_width ellispe_height "
                  "ellispe_theta ellipse_rel_density multi phase no_stations "
                  "stations t_window_start t_window_end Boots\n"
                  )

        from array_info import array

        a = array(st)
        event_time = a.eventtime()
        geometry = a.geometry()
        distances = a.distances(type="deg")
        mean_dist = np.mean(distances)
        stations = a.stations()
        no_stations = len(stations)
        sampling_rate = st[0].stats.sampling_rate
        stlo_mean, stla_mean = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
        evdp = st[0].stats.sac.evdp
        evlo = st[0].stats.sac.evlo
        evla = st[0].stats.sac.evla
        t_min = window[0]
        t_max = window[1]
        Target_phase_times, time_header_times = a.get_predicted_times(phase)

        # the traces need to be trimmed to the same start and end time
        # for the shifting and clipping traces to work (see later).
        min_target = int(np.nanmin(Target_phase_times, axis=0)) + (t_min)
        max_target = int(np.nanmax(Target_phase_times, axis=0)) + (t_max)

        stime = event_time + min_target
        etime = event_time + max_target


        # trim the stream
        # Normalise and cut seismogram around defined window
        st = st.copy().trim(starttime=stime, endtime=etime)
        st = st.normalize()

        # get predicted slownesses and backazimuths
        predictions = a.pred_baz_slow(phases=[phase], one_eighty=True)

        # find the line with the predictions for the phase of interest
        row = np.where((predictions == phase))[0]

        (
            P,
            S,
            BAZ,
            PRED_BAZ_X,
            PRED_BAZ_Y,
            PRED_AZ_X,
            PRED_AZ_Y,
            DIST,
            TIME,
        ) = predictions[row, :][0]
        PRED_BAZ_X = float(PRED_BAZ_X)
        PRED_BAZ_Y = float(PRED_BAZ_Y)
        S = float(S)
        BAZ = float(BAZ)

        name = (
            str(event_time.year)
            + f"{event_time.month:02d}"
            + f"{event_time.day:02d}"
            + "_"
            + f"{event_time.hour:02d}"
            + f"{event_time.minute:02d}"
            + f"{event_time.second:02d}"
        )

        a1 = array(st)
        traces = a1.traces()
        shifted_traces = shift_traces(traces=traces,
                                      geometry=geometry,
                                      abs_slow=float(S),
                                      baz=float(BAZ),
                                      distance=float(mean_dist),
                                      centre_x=float(stlo_mean),
                                      centre_y=float(stla_mean),
                                      sampling_rate=sampling_rate)

        # predict arrival time
        arrivals = model.get_travel_times(
            source_depth_in_km=evdp,
            distance_in_degree=mean_dist,
            phase_list=[phase]
        )

        pred_time = arrivals[0].time

        # get point of the predicted arrival time
        pred_point = int(sampling_rate * (pred_time - min_target))

        # get points to clip window
        point_before = int(pred_point + (t_min * sampling_rate))
        point_after = int(pred_point + (t_max * sampling_rate))

        # clip the traces
        cut_shifted_traces = shifted_traces[:, point_before:point_after]

        # get the min time of the traces
        min_time = pred_time + t_min

        no_clusters = np.amax(self.labels) + 1
        means_xy, means_baz_slow = self.cluster_means()
        bazs_std, slows_std, slow_xs_std, slow_ys_std, azs_std, mags_std = self.cluster_std_devs(pred_x=PRED_BAZ_X, pred_y=PRED_BAZ_Y)
        ellipse_areas = self.cluster_ellipse_areas(std_dev=2)
        ellipse_properties = self.cluster_ellipse_properties(std_dev=2)
        points_clusters = self.group_points_clusters()
        arrival_times = self.estimate_travel_times(traces=cut_shifted_traces,
                                                   tmin=min_time,
                                                   sampling_rate=sampling_rate,
                                                   geometry=geometry,
                                                   distance=mean_dist,
                                                   pred_x=PRED_BAZ_X,
                                                   pred_y=PRED_BAZ_Y)


        # Option to filter based on ellipse size or vector deviation
        if Filter == True:
            try:

                distances = distance.cdist(
                    np.array([[PRED_BAZ_X, PRED_BAZ_Y]]), means_xy, metric="euclidean"
                )
                number_arrivals_slow_space = np.where(distances < slow_vec_error)[
                    0
                ].shape[0]

                number_arrivals = number_arrivals_slow_space
            except:
                multi = "t"
                number_arrivals = 0

        elif Filter == False:
            number_arrivals = no_clusters

        else:
            print("Filter needs to be True or False")
            exit()

        if number_arrivals > 1:
            multi = "y"

        elif number_arrivals == 0:
            print("no usable arrivals, exiting code")
            # exit()
            multi = "t"
        elif number_arrivals == 1:
            multi = "n"

        else:
            print("something went wrong in error estimates, exiting")
            # exit()
            multi = "t"

        # make new line

        # set counter to be zero, this will be used to label the arrivals as first second etc.
        usable_arrivals = 0

        if no_clusters != 0:
            for i in range(no_clusters):

                # create label for the arrival
                # get information for that arrival
                baz_obs = means_baz_slow[i, 0]

                baz_diff = baz_obs - float(BAZ)

                slow_obs = means_baz_slow[i, 1]
                slow_diff = slow_obs - float(S)

                slow_x_obs = myround(means_xy[i, 0])
                slow_y_obs = myround(means_xy[i, 1])

                del_x_slow = slow_x_obs - PRED_BAZ_X
                del_y_slow = slow_y_obs - PRED_BAZ_Y

                az = np.degrees(np.arctan2(del_y_slow, del_x_slow))
                mag = np.sqrt(del_x_slow**2 + del_y_slow**2)

                if az < 0:
                    az +=360


                distance = np.sqrt(del_x_slow ** 2 + del_y_slow ** 2)

                baz_std_dev = bazs_std[i]
                slow_std_dev = slows_std[i]

                x_std_dev = slow_xs_std[i]
                y_std_dev = slow_ys_std[i]

                az_std_dev = azs_std[i]
                mag_std_dev = mags_std[i]

                error_ellipse_area = ellipse_areas[i]

                width = ellipse_properties[i, 1]
                height = ellipse_properties[i, 2]
                theta = ellipse_properties[i, 3]


                # relocated event location
                reloc_evla, reloc_evlo = relocate_event_baz_slow(evla=evla,
                                                                evlo=evlo,
                                                                evdp=evdp,
                                                                stla=stla_mean,
                                                                stlo=stlo_mean,
                                                                baz=baz_obs,
                                                                slow=slow_obs,
                                                                phase=phase,
                                                                mod='prem')


                times = arrival_times[i]
                mean_time = np.mean(times)
                time_diff = mean_time - pred_time
                times_std_dev = np.std(times)

                # if error_ellipse_area <= error_criteria_area and error_ellipse_area > 1.0:
                #     multi = 'm'

                if Filter == True:
                    if distance < slow_vec_error:

                        points_cluster = points_clusters[i]

                        tree = KDTree(
                            points_cluster, leaf_size=self.points.shape[0] * 1.5
                        )
                        points_rad = tree.query_radius(
                            points_cluster, r=epsilon, count_only=True
                        )
                        densities = points_rad / (np.pi * (epsilon ** 2))
                        mean_density = np.mean(densities)

                        # update the usable arrivals count
                        usable_arrivals += 1
                        name_label = name + "_" + str(usable_arrivals)

                        # define the newline to be added to the file
                        newline = (
                            f"{name_label} {evla:.2f} {evlo:.2f} {evdp:.2f} {reloc_evla:.2f} "
                            f"{reloc_evlo:.2f} {stla_mean:.2f} {stlo_mean:.2f} {S:.2f} {slow_obs:.2f} "
                            f"{slow_diff:.2f} {slow_std_dev:.2f} {BAZ:.2f} {baz_obs:.2f} {baz_diff:.2f} "
                            f"{baz_std_dev:.2f} {PRED_BAZ_X:.2f} {slow_x_obs:.2f} "
                            f"{del_x_slow:.2f} {x_std_dev:.2f} {PRED_BAZ_Y:.2f} {slow_y_obs:.2f} "
                            f"{del_y_slow:.2f} {y_std_dev:.2f} {az:.2f} {az_std_dev:.2f} {mag:.2f} {mag_std_dev:.2f} "
                            f"{mean_time:.2f} {pred_time:.2f} {time_diff:.2f} {times_std_dev:.2f} "
                            f"{error_ellipse_area:.2f} {width:.2f} {height:.2f} {theta:.2f} {mean_density:.2f} "
                            f"{multi} {phase} {no_stations} {','.join(stations)} "
                            f"{window[0]:.2f} {window[1]:.2f} {Boots}\n"
                        )

                        # there will be multiple lines so add these to this list.
                        newlines.append(newline)

                    else:
                        print(
                            "The error for this arrival is too large, not analysing this any further"
                        )
                        ## change the labels
                        updated_labels = np.where(self.labels == i, -1, self.labels)

                        newline = ""
                        newlines.append(newline)

                elif Filter == False:

                    points_cluster = points_clusters[i]

                    tree = KDTree(points_cluster, leaf_size=self.points.shape[0] * 1.5)
                    points_rad = tree.query_radius(
                        points_cluster, r=epsilon, count_only=True
                    )
                    densities = points_rad / (np.pi * (epsilon ** 2))
                    mean_density = np.mean(densities)

                    # update the usable arrivals count
                    usable_arrivals += 1
                    name_label = name + "_" + str(usable_arrivals)

                    # define the newline to be added to the file
                    newline = (
                        f"{name_label} {evla:.2f} {evlo:.2f} {evdp:.2f} {reloc_evla:.2f} "
                        f"{reloc_evlo:.2f} {stla_mean:.2f} {stlo_mean:.2f} {S:.2f} {slow_obs:.2f} "
                        f"{slow_diff:.2f} {slow_std_dev:.2f} {BAZ:.2f} {baz_obs:.2f} {baz_diff:.2f} "
                        f"{baz_std_dev:.2f} {PRED_BAZ_X:.2f} {slow_x_obs:.2f} "
                        f"{del_x_slow:.2f} {x_std_dev:.2f} {PRED_BAZ_Y:.2f} {slow_y_obs:.2f} "
                        f"{del_y_slow:.2f} {y_std_dev:.2f} {az:.2f} {az_std_dev:.2f} {mag:.2f} {mag_std_dev:.2f} "
                        f"{mean_time:.2f} {pred_time:.2f} {time_diff:.2f} {times_std_dev:.2f} "
                        f"{error_ellipse_area:.2f} {width:.2f} {height:.2f} {theta:.2f} {mean_density:.2f} "
                        f"{multi} {phase} {no_stations} {','.join(stations)} "
                        f"{window[0]:.2f} {window[1]:.2f} {Boots}\n"
                    )


                    # there will be multiple lines so add these to this list.
                    newlines.append(newline)

                else:
                    print("Filter needs to be True or False")
                    exit()

        else:
            newline = ""
            newlines.append(newline)

        ## Write to file!

        # now loop over file to see if I have this observation already
        found = False
        added = False  # just so i dont write it twice if i find the criteria in multiple lines
        ## write headers to the file if it doesnt exist
        line_list = []
        if os.path.exists(file_path):
            with open(file_path, "r") as Multi_file:
                for line in Multi_file:
                    if name in line and phase in line and f"{stla_mean:.2f}" in line:
                        print("name and phase and stla in line, replacing")
                        if added == False:
                            line_list.extend(newlines)
                            added = True
                        else:
                            print("already added to file")
                        found = True
                    else:
                        line_list.append(line)
        else:
            with open(file_path, "w") as Multi_file:
                Multi_file.write(header)
                line_list.append(header)

        if not found:
            print("name or phase or stla not in line. Adding to the end.")
            line_list.extend(newlines)
        else:
            pass

        with open(file_path, "w") as Multi_file2:
            Multi_file2.write("".join(line_list))


        return newlines
