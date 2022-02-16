import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from array_info import array

from sklearn.model_selection import GridSearchCV

def break_sub_arrays(st, min_stat, min_dist, spacing):
    """
    Given a stream of sac files with station location headers populated,
    break up the stations into sub arrays which meet the criteria of
    a minimum number of stations within a radius.

    Parameters
    ----------
    st : Obspy stream object
        It is assumed the traces in the stream object are SAC
        files with headers stla, stlo, stel populated.

    min_stat : int
        Minimum number of stations for each sub array to have.

    min_dist : float
        A radius in radians used to define the maximum neighborhood
        for the sub array.

    spacing : float
        Spacing in radians used to define the spacing between sub arrays.

    Returns
    -------
    final_centroids : 2D array of floats
        [lat, lon] points of the stations
                       used to break up sub arrays.
    lats_lons_use : 2D array of floats
        [lat, lon] of the stations which meet
        are identified as not being noise by DBSCAN.
    lats_lons_core : 2D array of floats
        the core points recovered from the
        cluter analysis.
    stations_use : list of strings
        Station names corresponding to the coordinates in
        lats_lons_use
    """
    from array_info import array

    array = array(st)
    stations = array.stations()
    geometry = array.geometry()
    lons = geometry[:, 0]
    lats = geometry[:, 1]

    ## dbscan to remove non-dense stations
    ## haversine matric wants lat_lon
    lats_lons_deg = np.array(list(zip(lats, lons)))
    lats_lons = np.array(list(zip(np.deg2rad(lats), np.deg2rad(lons))))

    # use DBSCAN

    db = DBSCAN(eps=min_dist, min_samples=min_stat, metric="haversine").fit(lats_lons)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    core_samples = db.core_sample_indices_

    lons_core = lons[core_samples]
    lats_core = lats[core_samples]

    ## Store the usable stations in a 2D array!
    stations_use = np.array(stations)[np.where(labels >= 0)[0]]
    lons_use = lons[np.where(labels >= 0)[0]]
    lats_use = lats[np.where(labels >= 0)[0]]
    lats_lons_use = np.array(list(zip(np.deg2rad(lats_use), np.deg2rad(lons_use))))

    # store lats and lons of core points
    lats_lons_core = np.array(list(zip(np.deg2rad(lats_core), np.deg2rad(lons_core))))

    # make a tree of these lats and lons
    tree = BallTree(
        lats_lons_core, leaf_size=lats_lons_core.shape[0] / 2, metric="haversine"
    )
    # just a test of how the radius query works
    sub_array_test = tree.query_radius(X=lats_lons_core, r=min_dist)

    # make a copy of the lats and lons of the core points as reference
    core_points_as_centroids = np.copy(lats_lons_core)

    # create list for the final centroids
    final_centroids = []
    while core_points_as_centroids.size != 0:
        # first get all the core points within 2 degrees of the first core point in the
        sub_array, distances = tree.query_radius(
            X=np.array([core_points_as_centroids[0]]), r=spacing, return_distance=True
        )

        # add the first point to the centroid list
        final_centroids.append(core_points_as_centroids[0])

        # for every value not within the spacing distance of the centroid,
        # apply a mask and keep them
        # i.e. remove all core points within the spacing distance of the
        # current centroid.

        for s in sub_array[0]:
            value = lats_lons_core[s]
            # row_mask = (core_points_as_centroids != value).all(axis=1)
            # print(row_mask)
            # core_points_as_centroids = core_points_as_centroids[row_mask, :]
            core_points_as_centroids =  core_points_as_centroids[np.logical_not(np.logical_and(core_points_as_centroids[:,0]==value[0],
                                                                 core_points_as_centroids[:,1]==value[1]))]


    return final_centroids, lats_lons_use, lats_lons_core, stations_use


def get_station_density_KDE(geometry):
    """
    Given a geometry, this function will calculate the density of the station distribution
    for each station. This can be used to weight the stacking or other uses the user can
    think of.

    Parameters
    ----------
    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    type : string
        Do you want distances in degrees (deg) or kilometres (km).

    Returns
    -------
    station_densities : numpy array of floats
        Natural log of densities.

    """
    # recover the longitudes and latitudes
    lons = geometry[:, 0]
    lats = geometry[:, 1]

    # create 2D array of lons and lats then
    # convert to radians
    data = np.vstack((lons, lats)).T
    data_rad = np.radians(data)

    # create learning algorithm parameters
    density_distribution = KernelDensity(kernel="cosine", metric="haversine")

    # use grid search cross-validation to optimize the bandwidth
    # cross validation involved taking random samples of the dataset and
    # using a score function to estimate the best fit of the model to
    # the data

    # search over bandwidths from 0.1 to 10
    params = {"bandwidth": np.logspace(-2, 2, 200)}
    grid = GridSearchCV(density_distribution, params)
    grid.fit(data_rad)

    # get the best model
    kde = grid.best_estimator_

    # get the ln(density) values
    station_densities = kde.score_samples(data_rad)

    return station_densities
