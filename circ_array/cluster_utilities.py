
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

        return np.around(covariance_matrices, 2)

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
