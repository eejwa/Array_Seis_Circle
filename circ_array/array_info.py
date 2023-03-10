import numba as jit
import numpy as np
from scipy.signal import hilbert
from obspy import UTCDateTime
from geo_sphere_calcs import deg_km_az_baz
from obspy.taup import TauPyModel
from utilities import myround, clip_traces

class array:
    # init method or constructor
    def __init__(self, stream):
        self.stream = stream

    def stations(self):
        """
        Function to return all the stations in the Obspy stream provided

        Parameters
        ----------
        stream : Obspy stream object)
            Stream object of SAC files with the headers
            stla,stlo and stel populated

        Returns
        -------
        stations : 1D list of strings
            List of strings of station names.
        """
        stations = []
        for tr in self.stream:
            stations.append(tr.stats.station)

        return stations


    def eventtime(self):
        """
        Function to recover dates and times from sac file and return an obspy
        date-time object.

        Parameters
        ----------
        stream : Obspy stream object
            stream of SAC files with nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec populated

        Returns
        -------
        event_time : Obspy datetime
            Object of the date stored.
        """

        Y = self.stream[0].stats.sac.nzyear
        JD = self.stream[0].stats.sac.nzjday
        H = self.stream[0].stats.sac.nzhour
        M = self.stream[0].stats.sac.nzmin
        S = self.stream[0].stats.sac.nzsec
        mS = self.stream[0].stats.sac.nzmsec

        event_time = UTCDateTime(
            year=Y, julday=JD, hour=H, minute=M, second=S, microsecond=mS
        )

        return event_time


    def geometry(
        self, return_center=False, distance="degrees", verbose="False", relative="False"
    ):
        """
        Collects array geometry information and returns an array of lon, lat and elevation.
        Method to calculate the array geometry and the center coordinates in km or degrees.

        Parameters
        ----------
        stream : Obspy Stream object
            Obspy stream of SAC files which have stla, stlo and stel headers populated.

        distance : string
            Defines how the distances are given, either 'km' or 'degrees'. Defaults to degrees.

        return_center : Bool
            If true, it will only return the centre lon, lat and height.

        relative : Bool
            If true, the station locations will be relative to the mean lon, lat and elevation.

        Returns
        -------

        geometry : 3D array of floats
            The geometry of the stations as 2d :class:`np.ndarray`
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km or degrees.

            If return_center is true, only the centre lon lat elev will be returned.
        """

        station_no = len(self.stream)
        geometry = np.empty((station_no, 3))

        # for each trace object in stream, get station
        # coordinates and add to geometry
        for i, tr in enumerate(self.stream):
            geometry[i, 0] = tr.stats.sac.stlo
            geometry[i, 1] = tr.stats.sac.stla
            try:
                geometry[i, 2] = tr.stats.sac.stel
            except:
                geometry[i, 2] = 0

        # get means of lon, lat, elevation
        center_x = geometry[:, 0].mean()
        center_y = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()

        if distance == "km":
            # convert to kilometres
            for i in range(station_no):
                x, y = util_geo_km(0, 0, geometry[i, 0], geometry[i, 1])
                geometry[i, 0] = x
                geometry[i, 1] = y

            # update the mean x,y values if
            center_x = geometry[:, 0].mean()
            center_y = geometry[:, 1].mean()
            center_h = geometry[:, 2].mean()
            # if relative, find distance from the centre
            if relative == "True":
                for i in range(station_no):
                    x, y = util_geo_km(center_x, center_y, geometry[i, 0], geometry[i, 1])
                    geometry[i, 0] = x
                    geometry[i, 1] = y
                    geometry[i, 2] -= center_h

        # if want relative in degrees just subtrace mean lat and lon
        elif distance == "degrees" and relative == "True":
            geometry[:, 0] -= center_x
            geometry[:, 1] -= center_y
            geometry[:, 2] -= center_h

        else:
            pass

        if return_center:
            return [center_x, center_y, center_h]
        else:
            return geometry


    def distances(self, type="deg"):
        """
        Given a stream, this function creates an array containing the epicentral distances for each of the stations

        Parameters
        ----------
        stream : Obspy stream object
            Stream containing SAC file with the gcarc and dist headers populated.

        type : string
            Do you want distances in degrees (deg) or kilometres (km).

        Returns
        -------
        distances : 1D numpy array of floats
            The epicentral distances of each station
            from the event.

        """

        distances = np.empty(len(self.stream))

        for i, tr in enumerate(self.stream):
            if type == "deg":
                distances[i] = tr.stats.sac.gcarc
            elif type == "km":
                distances[i] = tr.stats.sac.dist

            else:
                print("'type' needs to be either 'deg' or 'km'")
                exit()

        return distances



    def traces(self):
        """
        Given an obspy stream, this will return a 2D array of the waveforms

        Parameters
        ----------
        stream : Obspy stream object
            Stream containing SAC files.

        Returns
        -------
        Traces : 2D numpy array of floats
            Traces that were stored in the stream in the same order.

        """
        Traces = []

        st = clip_traces(self.stream)

        for i, tr in enumerate(st):
            Traces.append(list(tr.data))

        return np.array(Traces)


    def phase_traces(self):
        """
        Given an obspy stream, this will return a 2D array of the waveforms

        Parameters
        ----------
        stream : Obspy stream object
            Stream containing SAC files.

        Returns
        -------
        Phase_traces : 2D numpy array of complex floats.
            Stores the phase traces processed from the traces
            recorded at each station.
        """
        Phase_traces = []

        for i, tr in enumerate(self.stream):
            tr.data = tr.data.astype(float)
            hilbert_trace = np.array(hilbert(tr.data))
            Phase_traces.append(np.angle(hilbert_trace))

        return np.array(Phase_traces)




    def pred_baz_slow(self, phases, one_eighty=True):
        """
        Predicts the baz and horizontal slownesses of the given phases using the infomation in the Obspy stream.

        Parameters
        ----------
        stream : Obspy Stream object
            Must be in SAC format.

        phases : list of strings
            Phases for which the baz and horizontal slowness will be calculated

        Param one_eighty : Bool
            If there is more than one arrival with the same name, will it arrive from a backazimuth 180 degrees away (e.g. major arc for S2KS).

        Returns
        -------
        preds : list of strings
            list of: ["Phase", "Ray_parameter", "Backazimuth", "Backazimuth_X",
                      "Backazimuth_Y", "Azimuth_X", "Azimuth_Y", "Mean_Ep_Dist",
                      "Predicted_Travel_Time"]
        """

        sin = np.sin
        cos = np.cos
        radians = np.radians

        # need to get the center lon and lat of stations:
        center_lon, center_lat, center_elev = self.geometry(
            distance="degrees", return_center=True
            )

        # find event lon and lat
        st = self.stream
        evla = st[0].stats.sac.evla
        evlo = st[0].stats.sac.evlo
        evdp = st[0].stats.sac.evdp

        # find baz
        dist_deg, dist_km, az, baz = deg_km_az_baz(
            lat1=float(evla),
            lon1=float(evlo),
            lat2=float(center_lat),
            lon2=float(center_lon),
        )

        # use TauP to predict the slownesses
        model = TauPyModel(model="prem")

        tap_out = model.get_travel_times(
            source_depth_in_km=float(evdp),
            distance_in_degree=float(dist_deg),
            receiver_depth_in_km=0.0,
            phase_list=phases,
        )
        # baz = float(out['backazimuth'])
        distance_avg = float(dist_deg)

        # The orientation of the wavefront at the stations is just
        # 180 + baz:
        az_pred = baz + 180

        # baz for major arc phases will arrive with the opposite backazimuth.
        if baz < 180:
            other_baz = baz + 180
        else:
            other_baz = baz - 180

        preds = []

        # for each phase
        for i in range(len(phases)):
            temp_list = []
            # for each line in the predictions
            for x in range(len(tap_out)):
                # if it's the phase's prediction, save it to a list.
                if tap_out[x].name == phases[i]:

                    temp_list.append(
                        [
                            tap_out[x].name,
                            tap_out[x].ray_param_sec_degree,
                            baz,
                            tap_out[x].time,
                        ]
                    )

            # if one arrival
            if len(temp_list) == 1:

                # calculate azimuths and backazimuths
                slow_x_baz = myround(float(temp_list[0][1]) * sin(radians(baz)))
                slow_y_baz = myround(float(temp_list[0][1]) * cos(radians(baz)))

                slow_x_az = myround(float(temp_list[0][1]) * sin(radians(az_pred)))
                slow_y_az = myround(float(temp_list[0][1]) * cos(radians(az_pred)))

                # add to results list
                preds.append(
                    [
                        temp_list[0][0],
                        temp_list[0][1],
                        baz,
                        slow_x_baz,
                        slow_y_baz,
                        slow_x_az,
                        slow_y_az,
                        distance_avg,
                        temp_list[0][3],
                    ]
                )

            # if there is more than one arrival it could be a major arc phase
            # arriving from the opposite direction.
            # I assume there will only be two major arc arrivals...
            if len(temp_list) > 1:

                ## the first one should be arriving from the "correct" backazimuth ##
                slow_x_baz = myround(float(temp_list[0][1]) * sin(radians(baz)))
                slow_y_baz = myround(float(temp_list[0][1]) * cos(radians(baz)))

                slow_x_az = myround(float(temp_list[0][1]) * sin(radians(az_pred)))
                slow_y_az = myround(float(temp_list[0][1]) * cos(radians(az_pred)))

                preds.append(
                    [
                        temp_list[0][0],
                        temp_list[0][1],
                        baz,
                        slow_x_baz,
                        slow_y_baz,
                        slow_x_az,
                        slow_y_az,
                        distance_avg,
                        temp_list[0][3],
                    ]
                )

                if one_eighty:
                    # use the other backazimuth to do calculations
                    for i in np.arange(1, len(temp_list), 2):
                        slow_x_other_baz = myround(
                            float(temp_list[i][1]) * sin(radians(other_baz))
                        )
                        slow_y_other_baz = myround(
                            float(temp_list[i][1]) * cos(radians(other_baz))
                        )

                        # the azimuth should now be 180 from the predicted azimuth
                        # since we defined the azimith as 180 + baz, the 'other azimuth'
                        # is just the backazimuth.

                        slow_x_other_az = myround(
                            float(temp_list[i][1]) * sin(radians(baz))
                        )
                        slow_y_other_az = myround(
                            float(temp_list[i][1]) * cos(radians(baz))
                        )

                        preds.append(
                            [
                                "%s_Major" % temp_list[i][0],
                                temp_list[i][1],
                                other_baz,
                                slow_x_other_baz,
                                slow_y_other_baz,
                                slow_x_other_az,
                                slow_y_other_az,
                                distance_avg,
                                temp_list[i][3],
                            ]
                        )
                else:
                    # if you dont want to add 180 to backazimuth then the same baz and az are used
                    # with different slowness values
                    for i in range(1, len(temp_list)):
                        slow_x_baz = myround(float(temp_list[i][1]) * sin(radians(baz)))
                        slow_y_baz = myround(float(temp_list[i][1]) * cos(radians(baz)))

                        slow_x_az = myround(float(temp_list[i][1]) * sin(radians(az_pred)))
                        slow_y_az = myround(float(temp_list[i][1]) * cos(radians(az_pred)))

                        preds.append(
                            [
                                temp_list[i][0],
                                temp_list[i][1],
                                baz,
                                slow_x_baz,
                                slow_y_baz,
                                slow_x_az,
                                slow_y_az,
                                distance_avg,
                                temp_list[i][3],
                            ]
                        )
        preds = np.array(preds)

        return preds


    def get_t_header_pred_time(self, phase):
        """
        Gives a stream of SAC files and phase, it will return the header
        where the travel time predictions for that phase is stored.

        Parameters
        ----------
        stream : Obspy stream
            Stream of SAC files with the tn and ktn headers populated.

        phase : string
            Phase of interest (.e.g S).

        Returns
        -------
        Target_time_header : string
            The time header where the travel time predictions
            for the phase is stored.

        """

        # these are the header labels storing the phase
        #  names in SAC files.
        labels = ["kt0", "kt1", "kt2", "kt3", "kt4", "kt5", "kt6", "kt7", "kt8", "kt9"]

        #  initial header target label header
        Target_time_header = None
        # for every trace
        for x, trace in enumerate(self.stream):
            # for every label
            for K in labels:
                try:
                    # get the phase name from the file
                    phase_label = getattr(trace.stats.sac, K).strip()
                    # if the name matches the target phase
                    if phase_label == phase:
                        # replace the target header name
                        if Target_time_header == None:
                            Target_time_header = K.replace("k", "")
                except:
                    pass

        return Target_time_header


    # collect the predicted arrival times for each phase saved in the SAC header files.
    def get_predicted_times(self, phase):
        """
        Collect the predicted arrival times for all SAC files in the stream and return arrays for
        the predicted times for the target phase and all time headers. The min, max and average predicted
        times for the target phase will be returned.

        Parameters
        ----------
        stream : Obspy Stream Object
            Stream of SAC files with the time (tn) and labels (ktn) populated.

        phase : string
            The phase you are interested in analysing (e.g. SKS). Must be stored in the SAC headers tn and tkn.

        Returns
        -------
        Target_phase_times : 2D array of floats
            Predicted travel times for the target phase for each station in the array.
        time_header_times : 2D array of floats
            Array of the prediected travel times for all phases for each station in the array.

        """
        # these are the header labels storing the phase
        #  names in SAC files.
        labels = ["kt0", "kt1", "kt2", "kt3", "kt4", "kt5", "kt6", "kt7", "kt8", "kt9"]

        # list for the trave times of the
        # target phase
        Target_phase_times = []

        # get the header storing the travel times of
        # the target phase
        Target_time_header = self.get_t_header_pred_time(phase=phase)

        # make list of list of all the phase predicted times.
        time_header_times = [[] for i in range(10)]
        for x, trace in enumerate(self.stream):
            # get the distance of the station
            ep_dist = trace.stats.sac.gcarc

            # not all the traces will have the same phases arriving due to epicentral
            # distance changes
            phases_tn = []
            phases = []
            # get the phase names stored in the file
            for K in labels:
                try:
                    phase_label = getattr(trace.stats.sac, K).strip()
                    phases_tn.append([str(phase_label), str(K.replace("k", ""))])
                    phases.append(str(phase_label))
                except:
                    pass
                # check to see if it is the same as the phase:

            # append the predicted time
            Target_phase_times.append(getattr(trace.stats.sac, Target_time_header))

            # get the time values for each of the phases in the file and store them
            for c in range(len(phases_tn)):
                timeheader = phases_tn[c][1]
                try:
                    time_header_times[c].append(
                        [
                            float(getattr(trace.stats.sac, timeheader.strip())),
                            float(ep_dist),
                            phases_tn[c][0],
                        ]
                    )
                except:
                    pass

        return np.array(Target_phase_times, dtype=object), np.array(time_header_times, dtype=object)
