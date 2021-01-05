"""
This is a module which has utility functions for general use in array seismology:

    - myround: round number to closest value of given precision.

    - get_stations: get list of stations in the stream.

    - get_eventtime: get obspy dattime object of origin time.

    - get_geometry: get array of lon,lat,elev for each station in array.

    - get_distances: get array of epicentral distances from event to stations.

    - get_station_density_KDE: get KDE of station density for each station.

    - clip_traces: clip longer trace objects to be same length.

    - get_traces: get 2D array of traces from the stream.

    - get_phase_traces: get 2D array of instantaneous phase values from stream.

    - get_slow_baz: calculate slowness and backazimuth.

    - deg_km_az_baz: calculate distances and azimuths of two lon/lat coords.

    - pred_baz_slow: predict slowness and backazimuth of list of phases.

    - get_t_header_pred_time: get SAC header holding TT of target phase.

    - get_predicted_times: extract times from SAC files of labeled phases.

    - findpeaks_XY: get the top N peaks in an array.

    - findpeaks_Pol: recover top N peaks if using a polar coord system.

    - pick_tw: manually pick time window to conduct analysis in.
"""

import numpy as np
from obspy.signal.util import util_geo_km
from circ_beam import haversine_deg
from obspy.taup import TauPyModel
from obspy import UTCDateTime
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def myround(x, prec=2, base=0.05):
    """
    Rounds the number 'x' to the nearest 'base' with precision 'prec'

    Param: x (float)
    Description: number to be rounded
    Param: prec (int)
    Description: number of decimal places for the rounded number.
    Param: base (float)
    Description: the interval to be rounded nearest to.

    Return:
            The number rounded to the nearest 'base' value.
    """
    return round(base * round(float(x) / base), prec)


def get_stations(stream):
    """
    Function to return all the stations in the Obspy stream provided

    Param: stream (Obspy stream object)
    Description: Stream object of SAC files with the headers
                 stla,stlo and stel populated

    Return: List of strings of station names.
    """
    stations = []
    for tr in stream:
        stations.append(tr.stats.station)

    return stations


def get_eventtime(stream):
    """
    Function to recover dates and times from sac file and return an obspy
    date-time object.

    Param: stream (Obspy stream object)
    Description: stream of SAC files with nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec populated

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
        year=Y, julday=JD, hour=H, minute=M, second=S, microsecond=mS
    )

    return event_time


def get_geometry(
    stream, return_center=False, distance="degrees", verbose="False", relative="False"
):
    """
    Collects array geometry information and returns an array of lon, lat and elevation.
    Method to calculate the array geometry and the center coordinates in km or degrees.

    Param: stream (Obspy Stream object)
    Description: must be in SAC format and have stla, stlo and stel headers populated.

    Param: distance (string)
    Description:  defines how the distances are given, either 'km' or 'degrees'. Defaults to degrees.

    Param: return_center (Bool)
    Description: if true, it will only return the centre lon, lat and height.

    Param: relative (Bool)
    Description: If true, the station locations will be relative to the mean lon, lat and elevation.

    Returns:

    The geometry of the stations as 2d :class:`np.ndarray`
    The first dimension are the station indexes with the same order
    as the traces in the stream object. The second index are the
    values of [lat, lon, elev] in km or degrees.

    if return_center is true, only the centre lon lat elev will be returned.
    """

    station_no = len(stream)
    geometry = np.empty((station_no, 3))

    # for each trace object in stream, get station
    # coordinates and add to geometry
    for i, tr in enumerate(stream):
        geometry[i, 0] = tr.stats.sac.stlo
        geometry[i, 1] = tr.stats.sac.stla
        geometry[i, 2] = tr.stats.sac.stel

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


def get_distances(stream, type="deg"):
    """
    Given a stream, this function creates an array containing the epicentral distances for each of the stations

    Param: stream (Obspy stream object)
    Description: stream containing SAC file with the gcarc and dist headers populated.

    Param: type (string)
    Description: do you want distances in degrees (deg) or kilometres (km).

    Return:
        distances = numpy array of floats describing the epicentral distances of each station
                    from the event.

    """

    distances = np.empty(len(stream))

    for i, tr in enumerate(stream):
        if type == "deg":
            distances[i] = tr.stats.sac.gcarc
        elif type == "km":
            distances[i] = tr.stats.sac.dist

        else:
            print("'type' needs to be either 'deg' or 'km'")
            exit()

    return distances


def get_station_density_KDE(geometry):
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
    print(kde)

    print(kde.score_samples(data_rad))
    # get the ln(density) values
    station_densities = kde.score_samples(data_rad)

    return station_densities


def clip_traces(stream):
    """
    The traces in the stream may be of different length which isnt great for stacking etc.
    This function will trim the traces to the same length on the smallest trace size.

    Param: stream (Obspy stream object)
    Description: any stream object which have traces with data in them.

    Return:
        stream - Obspy stream with data of equal lengths.
    """
    import numpy as np

    stimes = []
    etimes = []

    for trace in stream:
        stimes.append(trace.stats.starttime)
        etimes.append(trace.stats.endtime)

    stream = stream.trim(starttime=np.max(stimes), endtime=np.amin(etimes))

    return stream


def get_traces(stream):
    """
    Given an obspy stream, this will return a 2D array of the waveforms

    Param: stream (Obspy stream object)
    Description: stream containing SAC files.

    Return:
        2D numpy array of floats describing the traces.
    """
    Traces = []

    st = clip_traces(stream)

    for i, tr in enumerate(st):
        Traces.append(list(tr.data))

    return np.array(Traces)


def get_phase_traces(stream):
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


def deg_km_az_baz(lat1, lon1, lat2, lon2):
    """
    Description: Function to return the ditances in degrees and km over a spherical Earth
                 with the backazimuth and azimuth.
                 Distances calculated using the haversine formula.

    Param: lat(1/2) (float)
    Description: latitude of point (1/2)

    Param: lon(1/2) (float)
    Description: longitude of point (1/2)

    Return:
           dist_deg: distance between points in degrees.
           dist_km: distance between points in km.
           az: azimuth at location 1 pointing to point 2.
           baz" backzimuth at location 2 pointing to point 1.
    """
    # use haversine formula to get distance in degrees and km
    R = 6371
    dist_deg = haversine_deg(lat1, lon1, lat2, lon2)
    dist_km = np.radians(dist_deg) * R

    az = np.degrees(
        np.arctan2(
            (np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))),
            np.cos(np.radians(lat1)) * np.sin(np.radians(lat2))
            - np.sin(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.cos(np.radians(lon2 - lon1)),
        )
    )
    #    baz=np.degrees(np.arctan2((np.sin(np.radians(lon1-lon2))*np.cos(np.radians(lat1))), np.cos(np.radians(lat2))*np.sin(np.radians(lat1)) - np.sin(np.radians(lat2))*np.cos(np.radians(lat1))*np.cos(np.radians(lon1-lon2)) ))
    dLon = np.radians(lon1 - lon2)

    y = np.sin(dLon) * np.cos(np.radians(lat1))
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(lat1)) - np.sin(
        np.radians(lat2)
    ) * np.cos(np.radians(lat1)) * np.cos(dLon)

    baz = np.arctan2(y, x)

    baz = np.degrees(baz)

    if baz < 0:
        baz = (baz + 360) % 360

    return dist_deg, dist_km, az, baz


def get_slow_baz(slow_x, slow_y, dir_type):
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

    # find slowness mag via pythaogoras
    slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
    # angle from trigonometry
    azimuth = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)

    # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
    baz = azimuth % -360 + 180

    azimuth = np.array(azimuth)
    baz = np.array(baz)

    # make baz positive if it's negative:
    baz[baz < 0] += 360
    azimuth[azimuth < 0] += 360

    if dir_type == "baz":
        return slow_mag, baz
    elif dir_type == "az":
        print("azimuth")
        return slow_mag, azimuth
    else:
        pass


def pred_baz_slow(stream, phases, one_eighty=True):
    """
    Predicts the baz and horizontal slownesses of the given phases using the infomation in the Obspy stream.

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
    center_lon, center_lat, center_elev = get_geometry(
        stream, distance="degrees", return_center=True
    )

    # find event lon and lat
    st = stream
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

    results = []

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
            results.append(
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

            results.append(
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

                    results.append(
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

                    results.append(
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
    results = np.array(results)

    return results


def get_t_header_pred_time(stream, phase):
    """
    Gives a stream of SAC files and phase, it will return the header
    where the travel time predictions for that phase is stored.

    Param: stream (Obspy stream)
    Description: stream of SAC files with the tn and ktn headers populated.

    Param: phase (string)
    Description: phase of interest

    Return:
        Target_time_header: string of the time header where the travel time predictions
                            for the phase is stored.

    """

    # these are the header labels storing the phase
    #  names in SAC files.
    labels = ["kt0", "kt1", "kt2", "kt3", "kt4", "kt5", "kt6", "kt7", "kt8", "kt9"]

    #  initial header target label header
    Target_time_header = None
    # for every trace
    for x, trace in enumerate(stream):
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
def get_predicted_times(stream, phase):
    """
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

    """
    # these are the header labels storing the phase
    #  names in SAC files.
    labels = ["kt0", "kt1", "kt2", "kt3", "kt4", "kt5", "kt6", "kt7", "kt8", "kt9"]

    # list for the trave times of the
    # target phase
    Target_phase_times = []

    # get the header storing the travel times of
    # the target phase
    Target_time_header = get_t_header_pred_time(stream=stream, phase=phase)

    # make list of list of all the phase predicted times.
    time_header_times = [[] for i in range(10)]
    for x, trace in enumerate(stream):
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

    return np.array(Target_phase_times), np.array(time_header_times)


def findpeaks_XY(Array, xmin, xmax, ymin, ymax, xstep, ystep, N=10):
    """
    Peak finding algorith for a 2D array of values. The peaks will be searched for
    within a range of points from a predicted arrival. Edited from stack overflow
    answer: https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array.

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

    Return:
        The top N peaks of the array of the format [[x,y]].
    """

    # define space
    steps_x = int(np.round((xmax - xmin) / xstep, decimals=0)) + 1
    steps_y = int(np.round((ymax - ymin) / ystep, decimals=0)) + 1
    xpoints = np.linspace(xmin, xmax, steps_x, endpoint=True)
    ypoints = np.linspace(ymin, ymax, steps_y, endpoint=True)

    neighborhood = generate_binary_structure(2, 3)

    # maximum filter will take the array 'Array'
    # For each point it will take the adjacent grid points and find the highest ones
    # In other words, it will find the local maxima.
    # local_max = maximum_filter(Array, 3)
    local_max = maximum_filter(Array, footprint=neighborhood) == Array
    # local_min = minimum_filter(Array_New, 3)

    background = Array == 0

    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # OK I'll try and find the gradient at each point then take the lowest values.

    detected_peaks = local_max ^ eroded_background

    y_peak, x_peak = np.where(detected_peaks == 1)

    # get top N values
    val_points = np.array([Array[y_peak, x_peak], y_peak, x_peak]).T

    val_points_sorted = np.copy(val_points)

    # sort arguments based on the first column
    # Just like Python, in that [::-1] reverses the array returned by argsort()
    #  [:N] gives the first N elements of the reversed list
    val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort()][::-1][:N]

    # get x and y locations of top N points
    val_x_points_sorted = val_points_sorted[:, 2]
    val_y_points_sorted = val_points_sorted[:, 1]

    # find this location in slowness space
    x_peaks_space = xmin + (x_peak * xstep)
    y_peaks_space = ymin + (y_peak * ystep)

    x_vals_peaks_space = xmin + (val_x_points_sorted * xstep)
    y_vals_peaks_space = ymin + (val_y_points_sorted * ystep)

    peaks_combined = np.array((x_peaks_space, y_peaks_space)).T
    peaks_combined_vals = np.array((x_vals_peaks_space, y_vals_peaks_space)).T

    return peaks_combined_vals


def findpeaks_Pol(Array, smin, smax, bmin, bmax, sstep, bstep, N=10):
    """
    Peak finding algorith for a 2D array of values. The peaks will be searched for
    within a range of points from a predicted arrival. This is edited for the polar
    coordinate search output.

    Param: Array (2-D numpy array of floats).
    Description: 2-D array of floats representing power or some other parameter.

    Param: smin (float)
    Description: Minumum horizontal slowness.

    Param: smax (float)
    Description: Maximum horizontal slowness.

    Param: bmin (float)
    Description: Minumum backazimuth.

    Param: bmax (float)
    Description: Maximum backazimuth.

    Param: step (float)
    Description: increments of slowness values.

    Param: btep (float)
    Description: increments of backazimuth values.

    Param: N (int)
    Description: The top N peaks will be returned.

    Return:
        The top N peaks of the array in the form of [baz,slow].
    """

    # define space
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
    local_max = maximum_filter(Array, footprint=neighborhood) == Array

    background = Array == 0

    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # OK I'll try and find the gradient at each point then take the lowest values.

    detected_peaks = local_max ^ eroded_background

    s_peak, b_peak = np.where(detected_peaks == 1)

    # get top N values
    val_points = np.array([Array[s_peak, b_peak], s_peak, b_peak]).T

    val_points_sorted = np.copy(val_points)

    # sort arguments based on the first column
    # Just like Python, in that [::-1] reverses the array returned by argsort()
    #  [:N] gives the first N elements of the reversed list
    val_points_sorted = val_points_sorted[val_points_sorted[:, 0].argsort()][::-1][:N]
    val_b_points_sorted = val_points_sorted[:, 2]
    val_s_points_sorted = val_points_sorted[:, 1]

    # this gives all the peaks in the array.
    b_peaks_space = bmin + (b_peak * bstep)
    s_peaks_space = smin + (s_peak * sstep)

    # get locations in the array
    b_vals_peaks_space = bmin + (val_b_points_sorted * bstep)
    s_vals_peaks_space = smin + (val_s_points_sorted * sstep)

    b_peaks_baz_vals = b_vals_peaks_space
    s_peaks_baz_vals = s_vals_peaks_space

    peaks_combined_vals = np.array((b_peaks_baz_vals, s_peaks_baz_vals)).T

    return peaks_combined_vals


# manually pick time window around phase
def pick_tw(stream, phase, tmin=150, tmax=150, align=False):
    """
    Given an Obspy stream of traces, plot a record section and allow a time window to be picked around the phases of interest.

    Param: stream (Obspy stream object)
    Description: Sac files only.

    Param: phase (string)
    Description: Phase of interest (e.g. SKS)

    Return:
        The selected time window as numpy array [window_start, window_end].
    """

    # define a function to record the location of the clicks
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

    # get the header with the times of the target phase in it
    Target_time_header = get_t_header_pred_time(stream=stream, phase=phase)

    # get the min and max predicted time of the phase at the array
    Target_phase_times, time_header_times = get_predicted_times(
        stream=stream, phase=phase
    )

    avg_target_time = np.mean(Target_phase_times)
    min_target_time = np.amin(Target_phase_times)
    max_target_time = np.amax(Target_phase_times)

    print(avg_target_time)
    print(min_target_time)
    print(max_target_time)

    # plot a record section and pick time window
    # Window for plotting record section
    win_st = float(min_target_time - tmin)
    win_end = float(max_target_time + tmax)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    event_time = get_eventtime(stream)
    stream_plot = stream.copy
    stream_plot = stream.trim(
        starttime=event_time + win_st, endtime=event_time + win_end
    )
    stream = stream.normalize()

    # plot each trace with distance
    for i, tr in enumerate(stream):

        dist = tr.stats.sac.gcarc
        # if you want to align them, subtract the times of the target phase
        if align == True:
            tr_plot = tr.copy().trim(
                starttime=event_time
                + (getattr(tr.stats.sac, Target_time_header) - tmin),
                endtime=event_time + (getattr(tr.stats.sac, Target_time_header) + tmax),
            )
            time = np.linspace(-tmin, tmax, int((tmin + tmax) * tr.stats.sampling_rate))
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
        plt.xlim(-tmin, tmax)

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
            print("t%s: No arrival" % i)

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


def write_to_file(filepath, st, peaks, prediction, phase, time_window):
    """
    Description:
        Function to write event and station information with slowness vector
        properties to a results file.

    Param: outfile (string)
    Description: Name and path of results file

    Param: st (Obspy stream object)
    Description: Stream object of SAC files assumed to have headers populated
                 as described in the README.

    Param: peaks (2D numpy array of floats)
    Description: 2D array of floats [[baz, slow]]
                 for the arrival locations.

    Param: prediction (2D numpy array of floats)
    Description: 2D numpy array of floats of the predicted arrival
                 in [[baz, slow]].

    Param: phase (string)
    Description: target phase (e.g. SKS)

    Param: time_window (1D numpy array of floats)
    Description: numpy array of floats describing the start and end
                  of time window in seconds.

    Return:
        Nothing.
    """

    import os

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    else:
        pass

    event_time = get_eventtime(st)
    geometry = get_geometry(st)
    distances = get_distances(st, type="deg")
    mean_dist = np.mean(distances)
    stations = get_stations(st)
    centre_x, centre_y = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
    sampling_rate = st[0].stats.sampling_rate
    evdp = st[0].stats.sac.evdp
    evla = st[0].stats.sac.evla
    evlo = st[0].stats.sac.evlo

    name = (
        str(event_time.year)
        + f"{event_time.month:02d}"
        + f"{event_time.day:02d}"
        + "_"
        + f"{event_time.hour:02d}"
        + f"{event_time.minute:02d}"
        + f"{event_time.second:02d}"
    )
    stat_string = ",".join(stations)
    newlines = []

    # make the line string
    for peak in peaks:
        baz_obs = peak[0]
        baz_pred = prediction[0]
        baz_diff = baz_obs - baz_pred

        slow_obs = peak[1]
        slow_pred = prediction[1]
        slow_diff = slow_obs - slow_pred
        print(
            name,
            evlo,
            evla,
            evdp,
            centre_x,
            centre_y,
            baz_pred,
            baz_obs,
            baz_diff,
            slow_pred,
            slow_obs,
            slow_diff,
            ",".join(stations),
            time_window[0],
            time_window[1],
            phase,
        )
        newline = (
            name
            + f" {evlo:.2f} {evla:.2f} {evdp:.2f} {centre_x:.2f} {centre_y:.2f} {baz_pred:.2f} {baz_obs:.2f} {baz_diff:.2f} {slow_pred:.2f} {slow_obs:.2f} {slow_diff:.2f} "
            + stat_string
            + f" {time_window[0]:.2f} {time_window[1]:.2f} "
            + phase
            + " \n"
        )
        # %(name, evlo, evla, evdp, centre_x, centre_y,  baz_pred, baz_obs, baz_diff, slow_pred, slow_obs, slow_diff, ','.join(stations), time_window[0], time_window[1], phase)
        # there will be multiple lines so add these to this list.
        newlines.append(newline)

    header = "name evlo evla evdp stlo_mean stla_mean pred_baz baz_obs baz_diff pred_slow slow_obs slow_diff stations start_window end_window phase \n"

    # now loop over file to see if I have this observation already
    found = False
    added = (
        False  # just so i dont write it twice if i find the criteria in multiple lines
    )
    ## write headers to the file if it doesnt exist
    line_list = []

    if os.path.exists(filepath):
        with open(filepath, "r") as Multi_file:
            for line in Multi_file:
                if name in line and phase in line and f"{centre_y:.2f}" in line:
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
        with open(filepath, "w") as Multi_file:
            Multi_file.write(header)
            line_list.append(header)
    if not found:
        print("name or phase or stla not in line. Adding to the end.")
        line_list.extend(newlines)
    else:
        pass

    with open(filepath, "w") as Multi_file2:
        Multi_file2.write("".join(line_list))


def break_sub_arrays(st, min_stat, min_dist):
    """
    Given a stream of sac files with station location headers populated,
    break up the stations into sub arrays which meet the criteria of
    a minimum number of stations within a radius.

    Param: st (Obspy stream object)
    Description: It is assumed the traces in the stream object are SAC
                 files with headers stla, stlo, stel populated.

    Param: min_stat (int)
    Description: minimum number of stations for each sub array to have.

    Param: min_dist (float)
    Description: A radius in degrees used to define the maximum neighborhood
                 for the sub array.

    Return:
        - final_centroids: 2D array of [lat, lon] points of the stations
                           used to break up sub arrays.
        - lats_lons_use: 2D array of [lat, lon] of the stations which meet
                         are identified as not being noise by DBSCAN.
        - lats_lons_core: 2D array of the core points recovered from the
                          cluter analysis.
        - stations_use: station names corresponding to the coordinates in
                        lats_lons_use
    """
    from sklearn.neighbors import BallTree
    from sklearn.cluster import DBSCAN

    stations = get_stations(st)
    geometry = get_geometry(st)
    lons = geometry[:, 0]
    lats = geometry[:, 1]

    ## dbscan to remove non-dense stations
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
            X=np.array([core_points_as_centroids[0]]), r=min_dist, return_distance=True
        )

        # add the first point to the centroid list
        final_centroids.append(core_points_as_centroids[0])

        for s in sub_array[0]:
            value = lats_lons_core[s]
            row_mask = (core_points_as_centroids != value).all(axis=1)
            core_points_as_centroids = core_points_as_centroids[row_mask, :]

    return final_centroids, lats_lons_use, lats_lons_core, stations_use
