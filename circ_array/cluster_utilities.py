
import numpy as np
from scipy.stats import circmean, circstd, circvar
import obspy

import circ_array as c

class cluster_utilities:

    """
    This class holds functions to give information about the clusters such as their mean,
    standard deviations, area, ellipse properties.
    """

    def __init__(self, labels, points):
        """
        All functions require the data points and their associated labels.
        Labels are given as output of clustering algorithms and are a list
        of integers. Each integer represents a cluster.

        Param: labels - 1D numpy array
        Description: contains the cluster labels for each datapoint
        e.g. labels = np.array([0, 0, 1, 1, 1, 0])

        Param: points - 2D numpy array
        Description: Contains datapoints used for the cluster analysis
        e.g. points = np.array([[0,0], [1,0], [1,2], [2,2], [3,2], [0,1]])

        """
        # assert(labels.shape == points.shape[0])
        self.labels = labels
        self.points = points


    def group_points_clusters(self):
        """
        From the points and labels, this function will split the points
        into n arrays (n being the number of clusters) and store them
        in an array.

        No inputs needed.

        Return:
            - points_cluters - 3D array where each row has a 2D array of points
                               for a particular cluster.
        """

        points_clusters = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]

            points_cluster = np.array([points_x, points_y]).T
            points_clusters.append(points_cluster)

        return points_clusters

    def eigsorted(self, cov):
        """
        Given a covariance matrix, calculate the eigenvalues and eigenvectors.

        Param: cov - 2D numpy array of floats
        Description: covariance matrix of an ndarray of points. Can be found from the np.cov function.

        Return:
            vals - eigenvalues in decending order.
            vecs - eigenvectors in decending order.
        """

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]  # create decending order
        # return eigenvalues and eigenvectors in decending order
        return vals[order], vecs[:, order]

    def cluster_means(self):
        """
        Given data points and cluster labels, return the mean slowness vector
        properties of each cluster.

        Param: points - 2D numpy array of floats.
        Description: Array of points used to find clusters.

        Param: labels - 1D numpy array of integers.
        Description: - Labels of each point for which cluster it is in.

        Param: rel_x - float
        Description: x component of slowness vector used to align traces for relative beamforming.

        Param: rel_y - float
        Description: y component of slowness vector used to align traces for relative beamforming.

        Return:
                means_xy - 2D array of the mean slow_x and slow_y for each cluster.

                means_baz_slow - 2D array of the meanbackazimuth and horizontal slowness
                                 for each cluster.
        """

        means_xy = []
        means_baz_slow = []

        for p in range(np.amax(self.labels) + 1):

            p_x, p_y = np.mean(self.points[np.where(self.labels == p)][:, 0]), np.mean(
                self.points[np.where(self.labels == p)][:, 1])

            mean_abs_slow, mean_baz = c.get_slow_baz(p_x, p_y, dir_type='az')

            mean_abs_slow = np.around(mean_abs_slow, 2)
            mean_baz = np.around(mean_baz, 1)

            means_xy.append(np.array([p_x, p_y]))
            means_baz_slow.append(np.array([mean_baz, mean_abs_slow]))

        return np.around(means_xy,2), np.around(means_baz_slow,2)


    def cluster_std_devs(self):
        """
        Given data points and cluster labels, return the standard deviations of
        backazimuth and horizontal slownesses of each cluster.

        Param: points - 2D numpy array of floats.
        Description: Array of points used to find clusters.

        Param: labels - 1D numpy array of integers.
        Description: - Labels of each point for which cluster it is in.

        Param: rel_x - float
        Description: x component of slowness vector used to align traces for relative beamforming.

        Param: rel_y - float
        Description: y component of slowness vector used to align traces for relative beamforming.

        Return:
                bazs_std - 1D array of the standard deviation for backazimuth values in each cluster.

                slows_std - 1D array of the standard deviation for horizontal slowness values
                            in each cluster.
        """

        bazs_std= []
        slows_std = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            abs_slows, bazs = c.get_slow_baz(points_x, points_y, dir_type='az')


            abs_slow_std_dev = np.around(np.std(abs_slows), 2)
            # stdev - use scipy's circstd
            baz_std_dev = np.around(np.degrees(circstd(np.radians(bazs))), 1)

            bazs_std.append(baz_std_dev)
            slows_std.append(abs_slow_std_dev)

        return np.around(bazs_std,2), np.around(slows_std,2)


    def covariance_matrices(self):
        """
        Given the labels and point locations, returns the covariance
        matrices for each cluster.

        Param: rel_x - float
        Description: x component of slowness vector used to align traces for relative beamforming.

        Param: rel_y - float
        Description: y component of slowness vector used to align traces for relative beamforming.

        Return:
            1D array of covariance matrices.
        """

        covariance_matrices = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]
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

        Param: std_dev - integer.
        Description: - standard deviation of error ellipse (typically 1,2, or 3).

        Return:
                ellipse_properties - 2D array of the error ellipse widths heights and thetas.
                                     [[i, width, height, theta], ..] for i in number_of_clusters.

        """


        ellipse_properties = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]

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

        Param: std_dev - integer.
        Description: - standard deviation of error ellipse (typically 1,2, or 3).

        Param: rel_x - float
        Description: x component of slowness vector used to align traces for relative beamforming.

        Param: rel_y - float
        Description: y component of slowness vector used to align traces for relative beamforming.

        Return:
                ellipse_areas - 1D array of the area for the error ellipse of the points in each cluster.

        """

        ellipse_areas = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            co_mat = np.cov(np.array([points_x, points_y]).T, rowvar=False)

            # calculate the area of the ellipse
            vals, vecs = self.eigsorted(co_mat)

            # calculate the width and height of the ellipse
            width, height = 2 * std_dev * np.sqrt(vals)

            Area_error_ellipse = (np.pi * (width/2) * (height/2))

            ellipse_areas.append(Area_error_ellipse)

        return np.around(ellipse_areas, 2)


    def cluster_baz_slow_95_conf(self, std_dev):
        """
        Given data points and cluster labels, return the 95% confidence range for baz and
        horizontal slowness in the cluster.

        Param: points - 2D numpy array of floats.
        Description: Array of points used to find clusters.

        Param: labels - 1D numpy array of integers.
        Description: - Labels of each point for which cluster it is in.

        Param: std_dev - integer.
        Description: - standard deviation of error ellipse (typically 1,2, or 3).

        Param: rel_x - float
        Description: x component of slowness vector used to align traces for relative beamforming.

        Param: rel_y - float
        Description: y component of slowness vector used to align traces for relative beamforming.

        Return:
                bazs_95_confidence - 2D array of floats. Each row contains the upper and lower
                                     bounds for the 95% backazimuth confidence intervals.
                                     Each row represents a clusters.

                slows_95_confidence - 2D array of floats. Each row contains the upper and lower
                                     bounds for the 95% horizontal slowness confidence intervals.
                                     Each row represents a clusters.

        """

        bazs_95_confidence = []
        slows_95_confidence = []

        for p in range(np.amax(self.labels) + 1):

            points_x, points_y = self.points[np.where(
                self.labels == p)][:, 0], self.points[np.where(self.labels == p)][:, 1]

            # points_x += rel_x
            # points_y += rel_y

            points_cluster = np.array([points_x, points_y]).T

            abs_slows, bazs = get_slow_baz_array(points_x, points_y, dir_type='az')

            mean_abs_slow = np.around(mean_abs_slow, 2)
            mean_baz = np.around(mean_baz, 1)

            abs_slows_sorted = np.sort(abs_slows)
            lower_abs_slow = np.around(np.percentile(abs_slows_sorted, 2.5), 2)
            upper_abs_slow = np.around(np.percentile(abs_slows_sorted, 97.5), 2)

            baz_diff = 360.0 - mean_baz

            # now rotate all angles by this so the mean baz is at 0/360
            rotated_bazs = (bazs + baz_diff) % 360

            # Now, using arctan2(), can find the angle difference frmo -pi to +pi
            diffs = np.degrees(np.arctan2(
                np.sin(np.radians(rotated_bazs)), np.cos(np.radians(rotated_bazs))))

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


    def create_newlines(self, st, file_path, phase, window, Boots, epsilon, slow_vec_error=3, Filter=False):
        """
        This function will create a list of lines with all relevant information to be stored in
        the results file. The function write_to_cluster_file() will write these lines to a new
        file and replace any of the lines with the same array location and target phase.

        Param: st (obspy stream object)
        Description: Obspy stream object of sac files with event, arrival time and
                     station headers populated.

        Param: file_path (string)
        Description: path to the results file to check the contents of.

        Param: phase (string)
        Description: target phase (e.g. SKS)

        Param: window (list of floats)
        Description: tmin and tmax describing the relative time window.

        Param: Boots (int)
        Description: number of bootstrap samples.

        Param: epsilon (float)
        Description: epsilon value used to find the clusters.

        Param: slow_vec_error (float)
        Description: maximum slowness vector deviation between the predicted
                     and observed arrival. Arrival with larger deviations will
                     be removed if Filter = True (below). Default is 3.

        Param: Filter (bool)
        Description: do you want to filter out the arrivals (default = False)

        Return:
            - newlines: list of strings of the contents to write to the results file.

        """

        from scipy.spatial import distance
        from sklearn.neighbors import KDTree
        import os

        newlines = []
        header = "Name evla evlo evdp stla_mean stlo_mean slow_pred slow_max slow_diff slow_std_dev baz_pred baz_max baz_diff baz_std_dev slow_x_pred slow_x_obs del_x_slow slow_y_pred slow_y_obs del_y_slow error_ellipse_area ellispe_width ellispe_height ellispe_theta ellipse_rel_density multi phase no_stations stations t_window_start t_window_end Boots \n"

        event_time = c.get_eventtime(st)
        geometry = c.get_geometry(st)
        distances = c.get_distances(st,type='deg')
        mean_dist = np.mean(distances)
        stations = c.get_stations(st)
        no_stations = len(stations)
        stlo_mean, stla_mean =  np.mean(geometry[:, 0]),  np.mean(geometry[:, 1])
        evdp = st[0].stats.sac.evdp
        evlo = st[0].stats.sac.evlo
        evla = st[0].stats.sac.evla

        # get predicted slownesses and backazimuths
        predictions = c.pred_baz_slow(stream=st, phases=[phase], one_eighty=True)

        # find the line with the predictions for the phase of interest
        row = np.where((predictions == phase))[0]

        P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]
        PRED_BAZ_X = float(PRED_BAZ_X)
        PRED_BAZ_Y = float(PRED_BAZ_Y)
        S = float(S)
        BAZ = float(BAZ)

        name = str(event_time.year) + f'{event_time.month:02d}' + f'{event_time.day:02d}'+ "_" + f'{event_time.hour:02d}' + f'{event_time.minute:02d}' + f'{event_time.second:02d}'

        no_clusters = np.amax(self.labels) + 1
        means_xy, means_baz_slow = self.cluster_means()
        bazs_std, slows_std = self.cluster_std_devs()
        ellipse_areas = self.cluster_ellipse_areas(std_dev=2)
        ellipse_properties = self.cluster_ellipse_properties(std_dev=2)
        points_clusters = self.group_points_clusters()
        # Option to filter based on ellipse size or vector deviation
        if Filter == True:
            try:

                distances = distance.cdist(np.array([[PRED_BAZ_X,PRED_BAZ_Y]]), means, metric="euclidean")
                number_arrivals_slow_space = np.where(distances < slow_vec_error)[0].shape[0]

                number_arrivals = number_arrivals_slow_space
            except:
                multi='t'
                number_arrivals = 0

        elif Filter == False:
            number_arrivals = no_clusters

        else:
            print('Filter needs to be True or False')
            exit()

        if number_arrivals > 1:
            multi = "y"

        elif number_arrivals == 0:
            print("no usable arrivals, exiting code")
            #exit()
            multi='t'
        elif number_arrivals == 1:
            multi = "n"

            with open(events_n_multi_path, 'a') as No_file:
                No_file.write("%s, %s \n" %(foldername, phase))
        else:
            print("something went wrong in error estimates, exiting")
            #exit()
            multi='t'

        # make new line
        usable_means = np.empty((number_arrivals,2))

        # set counter to be zero, this will be used to label the arrivals as first second etc.
        usable_arrivals = 0

        if no_clusters != 0:
            for i in range(no_clusters):

                # create label for the arrival
                # get information for that arrival
                baz_obs = means_baz_slow[i,0]

                baz_diff = baz_obs - float(BAZ)

                slow_obs = means_baz_slow[i,1]
                slow_diff = slow_obs - float(S)

                slow_x_obs = c.myround(means_xy[i,0])
                slow_y_obs = c.myround(means_xy[i,1])

                del_x_slow = slow_x_obs - PRED_BAZ_X
                del_y_slow = slow_y_obs - PRED_BAZ_Y

                distance = np.sqrt(del_x_slow**2 + del_y_slow**2)

                baz_std_dev = bazs_std[i]
                slow_std_dev = slows_std[i]

                error_ellipse_area = ellipse_areas[i]

                width = ellipse_properties[i,1]
                height = ellipse_properties[i,2]
                theta = ellipse_properties[i,3]


                # if error_ellipse_area <= error_criteria_area and error_ellipse_area > 1.0:
                #     multi = 'm'

                if Filter == True:
                    if distance < slow_vec_error:

                        usable_means[usable_arrivals] = np.array([slow_x_obs,slow_y_obs])

                        points_cluster = points_clusters[i]

                        tree = KDTree(points_cluster, leaf_size=self.points.shape[0]*1.5)
                        points_rad = tree.query_radius(points_cluster, r=epsilon, count_only=True)
                        densities = points_rad / (np.pi * (epsilon**2))
                        mean_density = np.mean(densities)

                        # update the usable arrivals count
                        usable_arrivals += 1
                        name_label = name + "_" + str(usable_arrivals)

                        # define the newline to be added to the file
                        newline = f"{name_label} {evla} {evlo} {evdp} {stla_mean} "\
                                  f"{stlo_mean} {str(S)} {str(slow_obs)} {str(slow_diff)} "\
                                  f"{str(slow_std_dev)} {str(BAZ)} {str(baz_obs)} {str(baz_diff)} "\
                                  f"{str(baz_std_dev)} {str(PRED_BAZ_X)} {str(slow_x_obs)} "\
                                  f"{str(del_x_slow)} {str(PRED_BAZ_Y)} {str(slow_y_obs)} "\
                                  f"{str(del_y_slow)} {str(error_ellipse_area)} {str(width)} "\
                                  f"{str(height)} {str(theta)} {str(mean_density)} {multi} "\
                                  f"{phase} {str(no_stations)} {','.join(stations)} {window[0]} "\
                                  f"{window[1]} {Boots} \n"

                        # there will be multiple lines so add these to this list.
                        newlines.append(newline)


                    else:
                        print('The error for this arrival is too large, not analysing this any further')
                        ## change the labels
                        labels = np.where(labels==i,-1,labels)


                        newline = ""
                        newlines.append(newline)

                elif Filter == False:


                    usable_means[usable_arrivals] = np.array([slow_x_obs, slow_y_obs])

                    points_cluster = points_clusters[i]


                    tree = KDTree(points_cluster, leaf_size=self.points.shape[0]*1.5)
                    points_rad = tree.query_radius(points_cluster, r=epsilon, count_only=True)
                    densities = points_rad / (np.pi * (epsilon**2))
                    mean_density = np.mean(densities)

                    # update the usable arrivals count
                    usable_arrivals += 1
                    name_label = name + "_" + str(usable_arrivals)

                    # define the newline to be added to the file
                    newline = f"{name_label} {evla:.2f} {evlo:.2f} {evdp:.2f} {stla_mean:.2f} "\
                              f"{stlo_mean:.2f} {S:.2f} {slow_obs:.2f} {slow_diff:.2f} "\
                              f"{slow_std_dev:.2f} {BAZ:.2f} {baz_obs:.2f} {baz_diff:.2f} "\
                              f"{baz_std_dev:.2f} {PRED_BAZ_X:.2f} {slow_x_obs:.2f} "\
                              f"{del_x_slow:.2f} {PRED_BAZ_Y:.2f} {slow_y_obs:.2f} "\
                              f"{del_y_slow:.2f} {error_ellipse_area:.2f} {width:.2f} "\
                              f"{height:.2f} {theta:.2f} {mean_density:.2f} {multi} "\
                              f"{phase} {no_stations} {','.join(stations)} {window[0]:.2f} "\
                              f"{window[1]:.2f} {Boots} \n"

                    # there will be multiple lines so add these to this list.
                    newlines.append(newline)

                else:
                    print('Filter needs to be True or False')
                    exit()

        else:
            newline = ""
            newlines.append(newline)


        ## Write to file!

        # now loop over file to see if I have this observation already
        found = False
        added = False # just so i dont write it twice if i find the criteria in multiple lines
        ## write headers to the file if it doesnt exist
        line_list = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as Multi_file:
                for line in Multi_file:
                    if name in line and phase in line and str(stla_mean) in line:
                        print("name and phase and stla in line, replacing")
                        if added == False:
                            line_list.extend(newlines)
                            added= True
                        else:
                            print('already added to file')
                        found = True
                    else:
                        line_list.append(line)
        else:
            with open(file_path, 'w') as Multi_file:
                Multi_file.write(header)
                line_list.append(header)

        if not found:
            print("name or phase or stla not in line. Adding to the end.")
            line_list.extend(newlines)
        else:
            pass

        with open(file_path, 'w') as Multi_file2:
            Multi_file2.write("".join(line_list))

        return newlines
