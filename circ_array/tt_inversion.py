import numpy as np 
from array_info import array as info
from utilities import closest_station
from sklearn.metrics.pairwise import haversine_distances

from scipy.linalg import lstsq as l_lstsq
from scipy.optimize import least_squares as nl_lstsq


class SlowVecInversion(): 
    def __init__(self, geometry, distances, ttimes):
        """
        Class to perform inversion from travel times at an array of stations
        to a slowness vector measurement. 

        Parameters
        ----------
        geometry : 2d arrat of floats
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
        self.centre_station, self.centre_station_index  = closest_station(centre = [np.mean(self.stlos), np.mean(self.stlos)], seis_array = self.geometry[:,:2])

        self.del_xs = self.stlos - self.centre_station[1]
        self.del_ys = self.stlas - self.centre_station[0]

        self.del_ttimes = self.ttimes - self.ttimes[self.centre_station_index]

    def circ_wave(self, slow_vec):
        """
        function to predict travel times for an array given a baz and slow.
        God i hope this works. 
        """

        baz, slow = slow_vec
        # relocate event from the centre of the array 
        lat2 = np.arcsin(
            (np.sin(self.centre_station[1]) * np.cos(self.dists[self.centre_station_index])) + (np.cos(self.centre_station[1]) * np.sin(self.dists[self.centre_station_index]) * np.cos(np.radians(baz)))
        )[0][0]
        lon2 = self.centre_station[0] + np.arctan2(
            np.sin(np.radians(baz)) * np.sin(self.dists[self.centre_station_index]) * np.cos(self.centre_station[1]), np.cos(self.dists[self.centre_station_index]) - np.sin(self.centre_station[1]) * np.sin(lat2)
        )[0][0]

        print(lat2,lon2)

        # calculate distances from new event to stations
        new_dists = haversine_distances(np.radians(self.geometry[:,:2]), [[lat2,lon2]])

        # get relative distances 
        rel_dist = new_dists - np.mean(new_dists)

        # multiply by slowness 
        times = np.degrees(rel_dist) * slow
        residual = self.traveltimes - times[:,0]

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
        """

        slow_vec = nl_lstsq(self.circ_wave, np.array(initial_slow_vec)).x


        return slow_vec


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

        p_pred = np.sqrt(p[0]**2 + p[1]**2)
        az_pred = np.degrees(np.arctan2(p[0], p[1]))  # * (180. / math.pi)
        baz_pred = (az_pred % -360) + 180

        if baz_pred < 0:
            baz_pred += 360

        return slow_vec, p_pred, baz_pred


