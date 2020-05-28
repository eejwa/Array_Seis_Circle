# import packages
import numpy as np
from obspy.signal.util import util_geo_km
from Circ_Beam import haversine_deg
from obspy.taup import TauPyModel
from obspy import UTCDateTime
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class Circ_Array:
    def __init__(self):
        help = """This is a class which implements curved wavefront correction for array seismology methods such as beamforming.
        For more information, see the documentation in the docs/ directory of this repository.
        """

    def myround(self, x, prec=2, base=.05):
        """
        Description: rounds the number 'x' to the nearest 'base' with precision 'prec'

        Param: x (float)
        Description: number to be rounded
        Param: prec (int)
        Description: number of decimal places for the rounded number.
        Param: base (float)
        Description: the interval to be rounded nearest to.

        Return:
                The number rounded to the nearest 'base' value.
        """
        return round(base * round(float(x)/base),prec)


    def get_stations(self,stream):
        """
        Function to return all the stations in the Obspy stream provided

        Param: stream (Obspy stream object)
        Description: Stream object of SAC files with the headers
                     stla,stlo and stel populated

        Return: List of stations.
        """
        stations=[]
        for tr in stream:
            stations.append(tr.stats.station)

        return stations


    def get_eventtime(self, stream):
        """
        function to recover dates and times from sac file and return an obspy
        date-time object.

        Param: stream (Obspy stream object)
        Description: stream of SAC files with nzyear, nzjday,nzhour,nzmin,nzsec,nzmsec populated

        Return:
            Obspy datetime object of the date stored.
        """

        Y = stream[0].stats.sac.nzyear
        JD = stream[0].stats.sac.nzjday
        H = stream[0].stats.sac.nzhour
        M = stream[0].stats.sac.nzmin
        S = stream[0].stats.sac.nzsec
        mS = stream[0].stats.sac.nzmsec

        event_time = UTCDateTime(
            year=Y, julday=JD, hour=H, minute=M, second=S, microsecond=mS)

        return event_time

    def get_geometry(self, stream, return_center=False, distance='degrees', verbose='False', relative='False'):
        """
        Collects array geometry information and returns an array of lon, lat and elevation.
        Method to calculate the array geometry and the center coordinates in km or degrees.

        Param: stream (Obspy Stream object)
        Description: must be in SAC format.

        Param: distance (string)
        Description:  defines how the distances are given, either 'km' or 'degrees'. Defaults to degrees.

        Param: return_center (Bool)
        Description: if true, it will only return the centre lon, lat and height.

        Param: relative (Bool)
        Description: If true, the station locations will be relative to the mean lon, lat and elevation.

        Returns the geometry of the stations as 2d :class:`np.ndarray`
        The first dimension are the station indexes with the same order
        as the traces in the stream object. The second index are the
        values of [lat, lon, elev] in km or degrees.

        if return_center is true, only the centre lon lat elev will be returned.
        """
        station_no = len(stream)
        geometry = np.empty((station_no, 3))

        for i,tr in enumerate(stream):
            geometry[i, 0] = tr.stats.sac.stlo
            geometry[i, 1] = tr.stats.sac.stla
            geometry[i, 2] = tr.stats.sac.stel

        center_x = geometry[:, 0].mean()
        center_y = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()

        if distance == "km":

            for i in range(station_no):
                x, y = util_geo_km(0, 0,
                                   geometry[i, 0], geometry[i, 1])
                geometry[i, 0] = x
                geometry[i, 1] = y

            # update the mean x,y values if
            # wanted i =n km
            center_x = geometry[:, 0].mean()
            center_y = geometry[:, 1].mean()
            center_h = geometry[:, 2].mean()
            if relative == 'True':
                for i in range(station_no):
                    x, y = util_geo_km(center_x, center_y,
                                       geometry[i, 0], geometry[i, 1])
                    geometry[i, 0] = x
                    geometry[i, 1] = y
                    geometry[i, 2] -= center_h

        elif distance == "degrees" and relative == 'True':
                geometry[:, 0] -= center_x
                geometry[:, 1] -= center_y
                geometry[:, 2] -= center_h

        else:
            pass

        if return_center:
            return [center_x, center_y, center_h]
        else:
            return geometry

    def get_distances(self, stream, type = 'deg'):
        """
        Given a stream, this function creates an array containing the epicentral distances for each of the stations

        Param: stream (Obspy stream object)
        Description: stream containing SAC file with the gcarc and dist headers populated.

        Param: type (string)
        Description: do you want distances in degrees (deg) or kilometres (km).

        Return:
            distances = numpy array of floats describing the epicentral distances.

        """

        distances = np.empty(len(stream))

        for i,tr in enumerate(stream):
            if type=='deg':
                distances[i] = tr.stats.sac.gcarc
            elif type=='km':
                distances[i] = tr.stats.sac.dist

            else:
                print("'type' needs to be either 'deg' or 'km'")
                exit()

        return distances


    def get_station_density_KDE(self, geometry):
        """
        Given a geometry, this function will calculate the density of the station distribution
        for each station. This can be used to weight the stacking or other uses the user can
        think of.

        Param: geometry (2D array of floats)
        Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

        Param: type (string)
        Description: do you want distances in degrees (deg) or kilometres (km).

        Return:
            station_densities = numpy array of natural log of densities.

        """

        lons = geometry[:,0]
        lats = geometry[:,1]

        data = np.vstack((lons,lats)).T
        data_rad = np.radians(data)

        # create learning algorithm parameters
        density_distribution = KernelDensity(kernel='cosine', metric='haversine')


        # use grid search cross-validation to optimize the bandwidth
        # cross validation involved taking random samples of the dataset and
        # using a score function to estimate the best fit of the model to
        # the data

        # search over bandwidths from 0.1 to 10
        params = {'bandwidth': np.logspace(-2, 2, 200)}
        grid = GridSearchCV(density_distribution, params)
        grid.fit(data_rad)

        # get the best model
        kde = grid.best_estimator_
        print(kde)

        print(kde.score_samples(data_rad))
        # density = np.exp(kde.score_samples(data_rad))
        station_densities = kde.score_samples(data_rad)

        return station_densities


    def clip_traces(self,stream):
        """
        The traces in the stream may be of different length which isnt great for stacking etc.
        This function will trim the traces to the same length on the smallest trace size.

        Param: stream (Obspy stream object)
        Description: any stream object which have traces with data in them.

        Return:
            stream - Obspy stream with data of equal lengths.
        """

        shapes = []

        for trace in stream:
            shapes.append(trace.data.shape)

        # get the minimum
        min_length = np.amin(shapes)

        for tr in stream:
            tr.data = tr.data[:min_length]

        return stream

    def get_traces(self,stream):
        """
        Given an obspy stream, this will return a 2D array of the waveforms

        Param: stream (Obspy stream object)
        Description: stream containing SAC files.

        Return:
            2D numpy array of floats describing the traces.
        """
        Traces = []

        st = self.clip_traces(stream)

        for i, tr in enumerate(st):
            Traces.append(list(tr.data))

        return np.array(Traces)

    def get_phase_traces(self,stream):
        """
        Given an obspy stream, this will return a 2D array of the waveforms

        Param: stream (Obspy stream object)
        Description: stream containing SAC files.

        Return:
            2D numpy array of floats describing the traces.
        """
        Phase_traces = []

        for i, tr in enumerate(stream):
            tr.data = tr.data.astype(float)
            hilbert_trace = np.array(hilbert(tr.data))
            Phase_traces.append(np.angle(hilbert_trace))

        return np.array(Phase_traces)

    def deg_km_az_baz(self, lat1, lon1, lat2, lon2):
        """
        Description: function to return the ditances in degrees and km over a spherical Earth
                     with the backazimuth and azimuth.
                     Distances calculated using the haversine formula.

        Param: lat(1/2): float
                       : latitude of point (1/2)

        Param: lon(1/2): float
                       : longitude of point (1/2)

        Return:
               dist_deg: distance between points in degrees.
               dist_km: distance between points in km.
               az: azimuth at location 1 pointing to point 2.
               baz" backzimuth at location 2 pointing to point 1.
        """
        R = 6371
        dist_deg = haversine_deg(lat1, lon1, lat2, lon2)
        dist_km = np.radians(dist_deg) * R

        az = np.degrees(np.arctan2((np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))), np.cos(np.radians(lat1)) *
                                   np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))))
    #    baz=np.degrees(np.arctan2((np.sin(np.radians(lon1-lon2))*np.cos(np.radians(lat1))), np.cos(np.radians(lat2))*np.sin(np.radians(lat1)) - np.sin(np.radians(lat2))*np.cos(np.radians(lat1))*np.cos(np.radians(lon1-lon2)) ))
        dLon = np.radians(lon1 - lon2)

        y = np.sin(dLon) * np.cos(np.radians(lat1))
        x = np.cos(np.radians(lat2)) * np.sin(np.radians(lat1)) - \
            np.sin(np.radians(lat2)) * np.cos(np.radians(lat1)) * np.cos(dLon)

        baz = np.arctan2(y, x)

        baz = np.degrees(baz)

        if baz < 0:
            baz = (baz + 360) % 360

        return dist_deg, dist_km, az, baz

    def get_slow_baz(self, slow_x, slow_y, dir_type):
        """
        Description:
            Returns the backazimuth and slowness magnitude of a slowness vector given its x and y components.

        Param: slow_x (array of floats)
        Description: X component of slowness vector.

        Param: slow_y (array of floats)
        Description: Y component of slowness vector.

        Param: dir_type (string)
        Description: how do you want the direction to be measured, backazimuth (baz) or azimuth (az).

        Return: slowness magnitude and baz/az value.
        """

        slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
        azimut = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)

        # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
        baz = azimut % -360 + 180

        # make baz positive if it's negative:
        baz[baz < 0] += 360
        azimut[azimut < 0] += 360

        if dir_type == "baz":
            return slow_mag, baz
        elif dir_type == "az":
            print("azimuth")
            return slow_mag, azimut
        else:
            pass


    def pred_baz_slow(self, stream, phases, one_eighty=True):
        """
        Description: Predicts the baz and horizontal slownesses of the given phases using the infomation in the Obspy stream.

        Param: stream (Obspy Stream object)
            Description: must be in SAC format.
        Param: phases (list of strings)
            Description: phases for which the baz and horizontal slowness will be calculated
        Param one_eighty (Bool)
            Description: if there is more than one arrival with the same name, will it arrive from a backazimuth 180 degrees away (e.g. major arc for S2KS).

        Return: list of: ["Phase", "Ray_parameter", "Backazimuth", "Backazimuth_X",
                          "Backazimuth_Y", "Azimuth_X", "Azimuth_Y", "Mean_Ep_Dist",
                          "Predicted_Travel_Time"]
        """

        sin = np.sin
        cos = np.cos
        radians = np.radians

        # need to get the center lon and lat of stations:
        center_lon, center_lat, center_elev = self.get_geometry(stream, distance="degrees", return_center=True)

        # find event lon and lat
        st = stream
        evla = st[0].stats.sac.evla
        evlo = st[0].stats.sac.evlo
        evdp = st[0].stats.sac.evdp

        # find baz
        dist_deg, dist_km, az, baz = self.deg_km_az_baz(lat1=float(evla), lon1=float(
            evlo), lat2=float(center_lat), lon2=float(center_lon))

        # use TauP to predict the slownesses
        model = TauPyModel(model="prem")

        tap_out = model.get_travel_times(source_depth_in_km=float(
            evdp), distance_in_degree=float(dist_deg), receiver_depth_in_km=0.0, phase_list=phases)
        # baz = float(out['backazimuth'])
        distance_avg = float(dist_deg)

        # The orientation of the wavefront at the stations is just
        # 180 + baz:
        az_pred = baz + 180
        # baz for other SKKS, which bounces on the opposite side of the outer core.

        if baz < 180:
            other_baz = baz + 180
        else:
            other_baz = baz - 180

        results = []

        # for each phase
        for i in range(len(phases)):
            temp_list = []
            # for each line in the predictions
            for x in range(len(tap_out)):
                # if it's the phase's prediction, save it to a file.
                if tap_out[x].name == phases[i]:

                    temp_list.append(
                        [tap_out[x].name, tap_out[x].ray_param_sec_degree, baz, tap_out[x].time])

            if len(temp_list) == 1:

                slow_x_baz = self.myround(float(temp_list[0][1]) * sin(radians(baz)))
                slow_y_baz = self.myround(float(temp_list[0][1]) * cos(radians(baz)))

                slow_x_az = self.myround(float(temp_list[0][1]) * sin(radians(az_pred)))
                slow_y_az = self.myround(float(temp_list[0][1]) * cos(radians(az_pred)))

                results.append([temp_list[0][0], temp_list[0][1], baz,
                                slow_x_baz, slow_y_baz, slow_x_az, slow_y_az, distance_avg, temp_list[0][3]])

            # The different phases behave differently, e.g. SKKS can have phases arriving at opposite bazs
            # This will mean that this will have to be edited with different "if" statements
            # I assume there will only be two SKKS arrivals...
            if len(temp_list) > 1:
                ## the first one should be arriving from the "correct" backazimuth ##

                slow_x_baz = self.myround(float(temp_list[0][1]) * sin(radians(baz)))
                slow_y_baz = self.myround(float(temp_list[0][1]) * cos(radians(baz)))

                slow_x_az = self.myround(float(temp_list[0][1]) * sin(radians(az_pred)))
                slow_y_az = self.myround(float(temp_list[0][1]) * cos(radians(az_pred)))

                results.append([temp_list[0][0], temp_list[0][1], baz,
                                slow_x_baz, slow_y_baz, slow_x_az, slow_y_az, distance_avg, temp_list[0][3]])

                if one_eighty:
                    for i in np.arange(1, len(temp_list),2):
                        slow_x_other_baz = self.myround(float(temp_list[i][1]) * sin(radians(other_baz)))
                        slow_y_other_baz = self.myround(float(temp_list[i][1]) * cos(radians(other_baz)))

                        # the azimuth should now be 180 from the predicted azimuth
                        # since we defined the azimith as 180 + baz, the 'other azimuth'
                        # is just the backazimuth.

                        slow_x_other_az = self.myround(float(temp_list[i][1]) * sin(radians(baz)))
                        slow_y_other_az = self.myround(float(temp_list[i][1]) * cos(radians(baz)))

                        results.append(["%s_Major" %temp_list[i][0], temp_list[i][1], other_baz, slow_x_other_baz,
                                        slow_y_other_baz, slow_x_other_az, slow_y_other_az, distance_avg, temp_list[i][3]])
                else:
                    for i in range(1, len(temp_list)):
                        slow_x_baz = self.myround(float(temp_list[i][1]) * sin(radians(baz)))
                        slow_y_baz = self.myround(float(temp_list[i][1]) * cos(radians(baz)))

                        slow_x_az = self.myround(float(temp_list[i][1]) * sin(radians(az_pred)))
                        slow_y_az = self.myround(float(temp_list[i][1]) * cos(radians(az_pred)))

                        results.append([temp_list[i][0], temp_list[i][1], baz,
                                        slow_x_baz, slow_y_baz, slow_x_az, slow_y_az, distance_avg, temp_list[i][3]])
        results = np.array(results)

        return results

    def get_t_header_pred_time(self, stream, phase):
        """
        Description: gives a stream of SAC files and phase, it will return the header
                     where the travel time predictions for that phase is stored.

        Param: stream (Obspy stream)
        Description: stream of SAC files with the tn and ktn headers populated.

        Param: phase (string)
        Description: phase of interest

        Return:
            Target_time_header: string of the time header where the travel time predictions
                                for the phase is stored.

        """

        labels = ["kt0", "kt1", "kt2", "kt3", "kt4",
                  "kt5", "kt6", "kt7", "kt8", "kt9"]

        Target_time_header = None
        for x, trace in enumerate(stream):

            for K in labels:
                try:
                    phase_label = getattr(trace.stats.sac, K).strip()
                    if phase_label == phase:
                        if Target_time_header == None:
                            Target_time_header = K.replace("k", "")
                except:
                    pass

        return Target_time_header

    # collect the predicted arrival times for each phase saved in the SAC header files.
    def get_predicted_times(self,stream, phase):
        '''
        Collect the predicted arrival times for all SAC files in the stream and return arrays for
        the predicted times for the target phase and all time headers. The min, max and average predicted
        times for the target phase will be returned.

        Param: stream (Obspy Stream Object)
        Description: Stream of SAC files with the time (tn) and labels (ktn) populated.

        Param: phase (string)
        Description: The phase you are interested in analysing (e.g. SKS). Must be stored in the SAC headers tn and tkn.

        Returns:
        Target_phase_times - an array of the predicted travel times for the target phase for each station in the array.
        time_header_times - array of the prediected travel times for all phases for each station in the array.

        '''

        labels = ["kt0", "kt1", "kt2", "kt3", "kt4",
                  "kt5", "kt6", "kt7", "kt8", "kt9"]

        Target_phase_times = []

        Target_time_header = self.get_t_header_pred_time(stream=stream, phase=phase)

        # make list of list of all the phase predicted times.
        time_header_times = [[] for i in range(10)]
        for x, trace in enumerate(stream):

            ep_dist = trace.stats.sac.gcarc

            # not all the traces will have the same phases arriving due to epicentral
            # distance changes
            phases_tn = []
            phases = []
            for K in labels:
                try:
                    phase_label = getattr(trace.stats.sac, K).strip()
                    phases_tn.append([str(phase_label), str(K.replace("k", ""))])
                    phases.append(str(phase_label))
                except:
                    pass
                # check to see if it is the same as the phase:

            # append the predicted time and make it relative to the event time
            Target_phase_times.append(getattr(trace.stats.sac, Target_time_header))

            for c in range(len(phases_tn)):
                timeheader = phases_tn[c][1]
                try:
                    time_header_times[c].append([float(
                        getattr(trace.stats.sac, timeheader.strip())), float(ep_dist), phases_tn[c][0]])
                except:
                    pass

        avg_target_time = np.mean(Target_phase_times)
        min_target_time = np.amin(Target_phase_times, axis=0)
        max_target_time = np.amax(Target_phase_times, axis=0)

        return np.array(Target_phase_times), np.array(time_header_times)


    def findpeaks_XY(self, Array, xmin, xmax, ymin, ymax, xstep, ystep, N=10):
        '''
        Peak finding algorith for a 2D array of values. The peaks will be searched for
        within a range of points from a predicted arrival
        Param: Array (2-D numpy array of floats).
        Description: 2-D array of floats representing power or some other parameter.
        Param: xmin (float)
        Description: Minumum x point of the area to search for peaks.
        Param: sl_xmax (float)
        Description: Maximum x point of the area to search for peaks.
        Param: sl_ymin (float)
        Description: Minumum y point of the area to search for peaks.
        Param: sl_ymax (float)
        Description: Maximum y point of the area to search for peaks.
        Param: step (float)
        Description: increments of points in x/y axis used in the array.
        Param: N (int)
        Description: The top N peaks will be returned.

        Return: The top N peaks of the array of the format [x,y].
        '''

        import numpy as np
        from scipy.ndimage.filters import maximum_filter
        from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

        # define space for plot

        steps_x = int(np.round((xmax - xmin) / xstep, decimals=0)) + 1
        steps_y = int(np.round((ymax - ymin) / ystep, decimals=0)) + 1
        xpoints = np.linspace(xmin, xmax, steps_x, endpoint=True)
        ypoints = np.linspace(ymin, ymax, steps_y, endpoint=True)

        xpoints = xpoints
        ypoints = ypoints

        neighborhood = generate_binary_structure(2, 3)

        # maximum filter will take the array 'Array'
        # For each point it will take the adjacent grid points and find the highest ones
        # In other words, it will find the local maxima.
        # local_max = maximum_filter(Array, 3)
        local_max = maximum_filter(Array, footprint=neighborhood) == Array
        # local_min = minimum_filter(Array_New, 3)

        background = (Array == 0)

        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        # OK I'll try and find the gradient at each point then take the lowest values.

        detected_peaks = local_max ^ eroded_background

        y_peak, x_peak = np.where(detected_peaks == 1)

        # print(Array[y_peak,x_peak], y_peak, x_peak)

        #  To be worked on to get top N values
        val_points = np.array([Array[y_peak, x_peak], y_peak, x_peak]).T

        # print(vals_Points[:,0],vals_Points[:,1],vals_Points[:,2])
        # print(val_points.shape)
        # print(np.matrix(val_points))

        val_points_sorted = np.copy(val_points)

        # sort arguments based on the first column
        # Just like Python, in that [::-1] reverses the array returned by argsort()
        # [:N] gives the first N elements of the reversed list
        val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort(
        )][::-1][:N]

        val_x_points_sorted = val_points_sorted[:, 2]
        val_y_points_sorted = val_points_sorted[:, 1]

        x_peaks_space = xmin + (x_peak * xstep)
        y_peaks_space = ymin + (y_peak * ystep)

        x_vals_peaks_space = xmin + (val_x_points_sorted * xstep)
        y_vals_peaks_space = ymin + (val_y_points_sorted * ystep)

        peaks_combined = np.array((x_peaks_space, y_peaks_space)).T
        peaks_combined_vals = np.array((x_vals_peaks_space, y_vals_peaks_space)).T

        return peaks_combined_vals


    def findpeaks_Pol(self, Array, smin, smax, bmin, bmax, sstep, bstep, N=10):
        '''
        Peak finding algorith for a 2D array of values. The peaks will be searched for
        within a range of points from a predicted arrival
        Param: Array (2-D numpy array of floats).
        Description: 2-D array of floats representing power or some other parameter.
        Param: xmin (float)
        Description: Minumum x point of the area to search for peaks.
        Param: sl_xmax (float)
        Description: Maximum x point of the area to search for peaks.
        Param: sl_ymin (float)
        Description: Minumum y point of the area to search for peaks.
        Param: sl_ymax (float)
        Description: Maximum y point of the area to search for peaks.
        Param: step (float)
        Description: increments of points in x/y axis used in the array.
        Param: N (int)
        Description: The top N peaks will be returned.

        Return: The top N peaks of the array in the form of [baz,slow].
        '''

        import numpy as np
        from scipy.ndimage.filters import maximum_filter
        from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

        # define space for plot

        steps_s = int(np.round((smax - smin) / sstep, decimals=0)) + 1
        steps_b = int(np.round((bmax - bmin) / bstep, decimals=0)) + 1

        spoints = np.linspace(smin, smax, steps_s, endpoint=True)
        bpoints = np.linspace(bmin, bmax, steps_b, endpoint=True)

        spoints = spoints
        bpoints = bpoints

        neighborhood = generate_binary_structure(2, 3)

        # maximum filter will take the array 'Array'
        # For each point it will take the adjacent grid points and find the highest ones
        # In other words, it will find the local maxima.
        # local_max = maximum_filter(Array, 3)
        local_max = maximum_filter(Array, footprint=neighborhood) == Array
        # local_min = minimum_filter(Array_New, 3)

        background = (Array == 0)

        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        # OK I'll try and find the gradient at each point then take the lowest values.

        detected_peaks = local_max ^ eroded_background

        s_peak, b_peak = np.where(detected_peaks == 1)



        # print(Array[y_peak,x_peak], y_peak, x_peak)

        #  To be worked on to get top N values
        val_points = np.array([Array[s_peak, b_peak], s_peak, b_peak]).T

        # print(vals_Points[:,0],vals_Points[:,1],vals_Points[:,2])
        # print(val_points.shape)
        # print(np.matrix(val_points))

        val_points_sorted = np.copy(val_points)

        # sort arguments based on the first column
        # Just like Python, in that [::-1] reverses the array returned by argsort()
        # [:N] gives the first N elements of the reversed list
        val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort(
        )][::-1][:N]
        val_b_points_sorted = val_points_sorted[:, 2]
        val_s_points_sorted = val_points_sorted[:, 1]

        # this gives all the peaks in the array.
        b_peaks_space = bmin + (b_peak * bstep)
        s_peaks_space = smin + (s_peak * sstep)

        b_vals_peaks_space = bmin + (val_b_points_sorted * bstep)
        s_vals_peaks_space = smin + (val_s_points_sorted * sstep)

        b_peaks_baz_vals = b_vals_peaks_space
        s_peaks_baz_vals = s_vals_peaks_space

        peaks_combined_vals = np.array((b_peaks_baz_vals, s_peaks_baz_vals)).T

        return peaks_combined_vals

    # manually pick time window around phase
    def pick_tw(self,stream, phase, tmin=150, tmax=150, align = False):
        '''
        Given an Obspy stream of traces, plot a record section and allow a time window to be picked around the phases of interest.

        Param: stream (Obspy stream object)
        Description: Sac files only.

        Param: phase (string)
        Description: Phase of interest (e.g. SKS)

        Returns the selected time window as numpy array [window_start, window_end].
        '''



        def get_window(event):
            """
            Description:
                    For an event such as a mouse click, return the x location of two events.

            Param: event
            Description: when creating interactive figure, an event will be a mouse click or key board press or something.

            Returns:
                    X locations of first two events event.
            """
            ix = event.xdata
            print("ix = %f" % ix)
            window.append(ix)
            # print(len(window))
            if np.array(window).shape[0] == 2:
                fig.canvas.mpl_disconnect(cid)
                plt.close()

            return window

        Target_time_header = self.get_t_header_pred_time(stream=stream, phase=phase)

        Target_phase_times, time_header_times = self.get_predicted_times(
            stream=stream, phase=phase)

        avg_target_time = np.mean(Target_phase_times)
        min_target_time = np.amin(Target_phase_times)
        max_target_time = np.amax(Target_phase_times)

        # plot a record section and pick time window
        # Window for plotting record section
        win_st = float(min_target_time - tmin)
        win_end = float(max_target_time + tmax)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        event_time = self.get_eventtime(stream)
        stream_plot = stream.copy
        stream_plot = stream.trim(starttime=event_time + win_st,
                              endtime=event_time + win_end)
        stream = stream.normalize()

        for i, tr in enumerate(stream):
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
                except:
                    pass
            else:
                pass

            try:
                ax.plot(np.sort(t[:, 0].astype(float)), np.sort(
                    t[:, 1].astype(float)), color='C'+str(i), label=t[0, 2])
            except:
                print("t%s: No arrival" %i)


        deg = u"\u00b0"

        # plt.title('Record Section Picking Window | Depth: %s Mag: %s' %(stream[0].stats.sac.evdp, stream[0].stats.sac.mag))
        plt.ylabel('Epicentral Distance (%s)' % deg)
        plt.xlabel('Time (s)')
        plt.legend(loc='best')

        window = []
        # turn on event picking package thing.
        cid = fig.canvas.mpl_connect('button_press_event', get_window)

        print("BEFORE YOU PICK!!")
        print("The first click of your mouse will the the start of the window")
        print("The second click will the the end of the window")
        plt.show()

        return window
