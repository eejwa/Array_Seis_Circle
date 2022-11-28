import numpy as np 
from array_info import array as info
from utilities import closest_station
from sklearn.metrics.pairwise import haversine_distances
from geo_sphere_calcs import coords_lonlat_rad_bearing, predict_pierce_points


from scipy.linalg import lstsq as l_lstsq
from scipy.optimize import least_squares as nl_lstsq


class SlowVecInversion(): 
    def __init__(self, geometry, distances, ttimes):
        """
        Class to perform inversion from travel times at an array of stations
        to a slowness vector measurement. 

        Parameters
        ----------
        geometry : 2d array of floats
                : geometry of the array in [lon,lat,elevation]

        distances : 1d array of floats
                : epicentral distances in the same order as the geometry

        ttimes : 1d array of floats
            : absolute travel times in the same order as the geometry    
        
        Returns
        -------
            SlowVecInversion object
        """

        self.geometry = geometry
        self.distances = distances
        self.ttimes = ttimes
        self.stlos = self.geometry[:,0]
        self.stlas = self.geometry[:,1]
        self.centre_station, self.centre_station_index  = closest_station(centre = np.radians([np.mean(self.stlas), np.mean(self.stlos)]), 
                                                                          seis_array = np.radians(np.stack([self.stlas, self.stlos]).T))

        self.centre_station = np.degrees(self.centre_station)

        self.centre_station_lo = self.centre_station[1]
        self.centre_station_la = self.centre_station[0]
        self.centre_station_dist = self.distances[self.centre_station_index]

        self.del_xs = self.stlos - self.centre_station_lo
        self.del_ys = self.stlas - self.centre_station_la
        self.del_ttimes = self.ttimes - self.ttimes[self.centre_station_index]

    def circ_wave(self, slow_vec):
        """
        function to predict travel times for an array given a baz and slow.
        God i hope this works. 
        """

        baz, slow = slow_vec
        # relocate event from the centre of the array 
        lat2, lon2 = coords_lonlat_rad_bearing(lat1 = self.centre_station_la, 
                                               lon1 = self.centre_station_lo, 
                                               dist_deg = self.centre_station_dist, 
                                               brng = baz)

        # calculate distances from new event to stations
        new_dists = np.degrees(haversine_distances(np.radians(np.stack([self.stlas, self.stlos]).T), 
                                        [np.radians([lat2,lon2])]))

        # get relative distances 
        # print(new_dists)
        # print(self.centre_station_dist, np.mean(new_dists))
        rel_dist = new_dists - self.centre_station_dist
        # print(rel_dist)

        # multiply by slowness 
        times = rel_dist * slow
        residual = self.del_ttimes - times[:,0]

        return residual

    def invert_circ(self, initial_slow_vec):
        """
        Inverts travel times to a slowness vector using a curved 
        wavefront approximation.

        Parameters
        ----------
        initial_slow_vec : 1d numpy array of floats
                         : intial guess slowness vector of [baz, slow]

        Returns
        -------
        slow_vec : 1d numpy array of floats
                 : inverted slowness vector [baz, slow]  

        result : scipy non linear least squares object
               : the full result from the inversion where users can 
                 explore the results.       
        """

        slow_vec_result = nl_lstsq(self.circ_wave, np.array(initial_slow_vec), loss='arctan', jac='3-point')
        slow_vec = slow_vec_result.x

        return slow_vec, slow_vec_result


    def invert_plane(self):
        """
        Function to invert travel times to a slowness vector. 

        Parameters
        ----------


        Returns
        -------
        slow_vec : 1D array of floats
                 : horizontal slowness vector [px,py]

        p_pred : float 
               : predicted horizontal slowness

        baz_pred : float 
                 : predicted backazimuth 
        
        
        """

        
        A = np.array(list(zip(self.del_xs, self.del_ys)))
        slow_vec, res, rnk, s = l_lstsq(A, self.del_ttimes)

        p_pred = np.sqrt(slow_vec[0]**2 + slow_vec[1]**2)
        az_pred = np.degrees(np.arctan2(slow_vec[0], slow_vec[1]))  # * (180. / math.pi)
        baz_pred = (az_pred % -360) + 180

        if baz_pred < 0:
            baz_pred += 360

        return slow_vec, p_pred, baz_pred


