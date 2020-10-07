#/usr/bin/env python

from numba import jit, jit_module
import numpy as np

@jit(nopython=True, fastmath=True)
def coords_lonlat_rad_bearing(lat1, lon1, dist_deg, brng):
    '''
    returns the latitude and longitude of a new cordinate that is the defined distance away and
    at the correct azimuth from the starting point.

    Param: lat1 (float)
    Description: starting point latitiude.

    Param: lon1 (float)
    Description: starting point longitude.

    Param: dist_deg (float)
    Description:  distance from starting point in degrees.

    Param: brng (float)
    Description: angle from north describing the direction where the new coordinate is located.

    Return:
        latitude and longitude of the new cordinate.
    '''

    brng = np.radians(brng)  # convert bearing to radians
    d = np.radians(dist_deg)  # convert degrees to radians
    lat1 = np.radians(lat1)  # Current lat point converted to radians
    lon1 = np.radians(lon1)  # Current long point converted to radians

    lat2 = np.arcsin((np.sin(lat1) * np.cos(d)) +
                     (np.cos(lat1) * np.sin(d) * np.cos(brng)))
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d) *
                             np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    if lon2 > 180:
        lon2 -= 360
    elif lon2 < -180:
        lon2 += 360
    else:
        pass

    return lat2, lon2


@jit(nopython=True, fastmath=True)
def haversine_deg(lat1, lon1, lat2, lon2):
    '''
    Function to calculate the distance in degrees between two points on a sphere.

    Param: lat1 (float)
    Description: latitiude of point 1.

    Param: lat1 (float)
    Description: longitiude of point 1.

    Param: lat2 (float)
    Description: latitiude of point 2.

    Param: lon2 (float)
    Description: longitude of point 2.

    Return:
        Distance between two points in degrees.

    '''
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2))**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * (np.sin(dlon / 2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = np.degrees(c)
    return d


@jit(nopython=True, fastmath=True)
def get_slow_baz(slow_x, slow_y, dir_type):
    '''
    Returns the backazimuth and slowness magnitude of a slowness vector given its x and y components.

    Param: slow_x (float)
    Description: X component of slowness vector.

    Param: slow_y (float)
    Description: Y component of slowness vector.

    Param: dir_type (string)
    Description: how do you want the direction to be measured, backazimuth (baz) or azimuth (az).

    Return:
        slowness magnitude and baz/az value.
    '''
    slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
    azimut = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)
    # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
    baz = azimut % -360 + 180
    # res = list of results :D
    # make baz positive if it's negative:
    if baz < 0:
        baz += 360
    if azimut < 0:
        azimut += 360

    if dir_type == "baz":
        return slow_mag, baz
    elif dir_type == "az":
        return slow_mag, azimut
    else:
        pass


@jit(nopython=True, fastmath=True)
def get_slow_baz_array(slow_x, slow_y, dir_type):
    '''
    Returns the backazimuth and slowness magnitude of a slowness vector given its x and y components.

    Param: slow_x (float)
    Description: X component of slowness vector.

    Param: slow_y (float)
    Description: Y component of slowness vector.

    Param: dir_type (string)
    Description: how do you want the direction to be measured, backazimuth (baz) or azimuth (az).

    return: slowness magnitude and baz/az value.
    '''
    slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
    azimut = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)
    # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
    baz = azimut % -360 + 180
    # res = list of results :D
    # make baz positive if it's negative:
    baz[baz < 0] += 360
    azimut[azimut < 0] += 360

    if dir_type == "baz":
        return slow_mag, baz
    elif dir_type == "az":
        return slow_mag, azimut
    else:
        pass


# replace some of this with other functions
@jit(nopython=True)
def ARF_process_f_s_spherical(geometry, sxmin, sxmax, symin, symax, sstep, distance, fmin, fmax, fstep, scale):
    """
    Returns array transfer function as a function of slowness difference and
    frequency. It will also write to a file in xyz format! This will only work for SAC files as there
    is a Obspy function for other files already...

    Param: geometry (2D numpy array)
    Description: numpy array of [lon, lat,elevation] values for each station.

    Param: s[x/y](min/max) (float).
    Description: the min/max value of the slowness of the wavefront in x/y direction.

    Param: sstep (float)
    Description: slowness interval in x andy direction.

    Param: distance (float)
    Description: the distance of the event from the centre of the stations, this will be used to estimate the curvature
    of the wavefront.

    Param: fmin (float)
    Description: minimum frequency in signal.

    Param: fmax (float)
    Description maximum frequency in signal.

    Param: fstep (float)
    Description: frequency sample distance.

    Param: scale: (Bool)
    Description: if True, the values will be normalised between 0 and 1.

    Return:
        transff: 2D numpy array of power values in a slowness grid for the array response function.
        ARF_arr: 2D numpy array of [slow_x,slow_y,power] can be written to an xyz file for GMT.
    """

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space) + 1))
    nsy = int((np.round((symax - symin) / s_space) + 1))

    nf = int(np.ceil((fmax + fstep / 10. - fmin) / fstep))

    # make empty array for output.
    buff = np.zeros(nf)
    transff = np.empty((nsx, nsy))
    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    ARF_arr = np.zeros((nsx * nsy, 3))

    # do the processing... many nested loops...
    for i in range(slow_xs.shape[0]):
        for j in range(slow_ys.shape[0]):

            sx = slow_xs[int(i)]
            sy = slow_ys[int(j)]
            # get the slowness and backazimuth of the vector
            abs_slow = np.sqrt(sx**2 + sy**2)
            azimut = np.degrees(np.arctan2(sx, sy))  # * (180. / math.pi)

            # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
            baz = azimut % -360 + 180

            # make baz positive:
            if baz < 0:
                baz += 360

            # calculate a coordinate of an imaginary event
            # from the backzimuth and distance get a new earthquake location

            brng = np.radians(baz)  # convert bearing to radians
            d = np.radians(distance)  # convert degrees to radians
            # Current lat point converted to radians
            lat1 = np.radians(centre_y)
            # Current long point converted to radians
            lon1 = np.radians(centre_x)

            lat_new = np.arcsin((np.sin(lat1) * np.cos(d)) +
                                (np.cos(lat1) * np.sin(d) * np.cos(brng)))
            lon_new = lon1 + np.arctan2(np.sin(brng) * np.sin(d) *
                                        np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat_new))

            lat_new_deg = np.degrees(lat_new)
            lon_new_deg = np.degrees(lon_new)

            if lon_new_deg > 180:
                lon_new_deg -= 360
            elif lon_new_deg < -180:
                lon_new_deg += 360
            else:
                pass

            # calculate distances
            sta_distances = np.empty(len(geometry))

            for x in range(geometry.shape[0]):
                stla = float(geometry[x, 1])
                stlo = float(geometry[x, 0])

                dlat = np.radians(abs(stla - lat_new_deg))
                dlon = np.radians(abs(stlo - lon_new_deg))
                a = float((np.sin(dlat / 2))**2 + np.cos(np.radians(lat_new_deg))
                          * np.cos(np.radians(stla)) * (np.sin(dlon / 2))**2)
                c = float(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
                dist = float(np.degrees(c))

                sta_distances[int(x)] = float(dist)

            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                _sum = 0j
                for l in range(sta_distances.shape[0]):
                    _sum += np.exp(
                        # Note: the coords need to be in [x,y] i.e. for each coord, 0=x dist and 1=y dist from the centre of the array.
                        # the zero is the amplitude (i.e. the ARF has no amplitude only the distribution)
                        complex(0., (sta_distances[int(l)] * abs_slow) *
                                2 * np.pi * f))
                buff[int(k)] = abs(_sum) ** 2
            transff[i, j] = np.trapz(buff)  # cumtrapz(buff, dx=fstep)[-1]

            point = int(int(j) + int(slow_xs.shape[0] * i))
            ARF_arr[point] = np.array([sx, sy, np.trapz(buff)])

    # normalise the array response function!!!
    transff /= transff.max()
    ARF_arr[:, 2] /= ARF_arr[:, 2].max()

    return transff, ARF_arr


@jit(nopython=True, fastmath=True)
def calculate_time_shifts(traces, geometry, abs_slow, baz, distance, centre_x, centre_y, type='circ'):
    """
    Calculates the time delay for each station relative to the time the phase should arrive at the centre
    of the array. Will use either a plane or curved wavefront approximation.

    Param: traces (2D numpy array of floats)
    Description: a 2D numpy array containing the traces that the user wants to stack.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: abs_slow (float)
    Description: horizontal slowness you want to align traces over.

    Param: baz (float)
    Description: backazimuth you want to align traces over.

    Param: centre_x (float)
    Description: mean longitude.

    Param: centre_y (float)
    Description: mean latitude.

    Param: type (string)
    Description: will calculate either using a curved (circ) or plane (plane) wavefront.

    Return:
        times - numpy array of the arrival time for the phase at
                each station relative to the centre.

    """


    brng = np.radians(baz)  # convert bearing to radians
    d = np.radians(distance)  # convert degrees to radians
    lat1 = np.radians(centre_y)  # Current lat point converted to radians
    lon1 = np.radians(centre_x)  # Current long point converted to radians

    slow_x = abs_slow * np.sin(brng)
    slow_y = abs_slow * np.cos(brng)

    lat_new, lon_new = coords_lonlat_rad_bearing(
        lat1=centre_y, lon1=centre_x, dist_deg=distance, brng=baz)

    # create array for shifted traces
    shifts = np.zeros(traces.shape[0])
    times = np.zeros(traces.shape[0])

    for x in range(geometry.shape[0]):
        stla = float(geometry[int(x), 1])
        stlo = float(geometry[int(x), 0])

        x_rel = stlo - centre_x
        y_rel = stla - centre_y
        if type == 'circ':
            dist = haversine_deg(lat1=lat_new, lon1=lon_new, lat2=stla, lon2=stlo)

            # get the relative distance
            dist_rel = float(dist) - float(distance)

            # get the travel time for this distance
            dt = float(dist_rel) * float(abs_slow)

            # the correction will be dt *-1
            shift = float(dt) * -1

            shifts[int(x)] = shift
            times[int(x)] = dt

        elif type == 'plane':
            dt = (x_rel * slow_x) + (y_rel * slow_y)

            shift = float(dt)

            times[int(x)] = shift

            dt *= -1
            times[int(x)] = dt

    return shifts, times


@jit(nopython=True, fastmath=True)
def shift_traces(traces, geometry, abs_slow, baz, distance, centre_x, centre_y, sampling_rate, type='circ'):
    """
    shifts the traces using the predicted arrival times for a given backazimuth and slowness.

    Param: traces (2D numpy array of floats)
    Description: a 2D numpy array containing the traces that the user wants to stack.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: abs_slow (float)
    Description: horizontal slowness you want to align traces over.

    Param: baz (float)
    Description: backazimuth you want to align traces over.

    Param: centre_x (float)
    Description: mean longitude.

    Param: centre_y (float)
    Description:mean latitude.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: type (string)
    Description: will calculate either using a curved (circ) or plane (plane) wavefront.

    Return:
        shifted_traces - 2D numpy array of floats of the shifted traces.
    """

    brng = np.radians(baz)  # convert bearing to radians
    d = np.radians(distance)  # convert degrees to radians
    lat1 = np.radians(centre_y)  # Current lat point converted to radians
    lon1 = np.radians(centre_x)  # Current long point converted to radians

    lat_new, lon_new = coords_lonlat_rad_bearing(
        lat1=centre_y, lon1=centre_x, dist_deg=distance, brng=baz)

    # create array for shifted traces
    shifted_traces = np.zeros(traces.shape)

    for x in range(geometry.shape[0]):
        stla = float(geometry[int(x), 1])
        stlo = float(geometry[int(x), 0])

        dist = haversine_deg(lat1=lat_new, lon1=lon_new, lat2=stla, lon2=stlo)

        # get the relative distance
        dist_rel = float(dist) - float(distance)

        # get the travel time for this distance
        dt = float(dist_rel) * float(abs_slow)

        # the correction will be dt *-1
        shift = float(dt) * -1

        pts_shift = int(shift * sampling_rate)
        # shift the traces with numpy.roll()
        shifted_trace = np.roll(traces[int(x)], pts_shift)

        shifted_traces[int(x)] = shifted_trace

    return shifted_traces


@jit(nopython=True, fastmath=True)
def linear_stack_baz_slow(traces, sampling_rate, geometry, distance, slow, baz):
    """
    Function to stack the given traces along a backazimuth and horizontal slownes.

    Param: Traces (2D numpy array of floats)
    Description: a 2D numpy array containing the traces that the user wants to stack.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: Distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: slow (float)
    Description: horizontal slowness you want to align traces over.

    Param: baz (float)
    Description: backazimuth you want to align traces over.

    Return:
    lin_stack - numpy array of the stacked waveform.
    """

    ntrace = traces.shape[0]

    centre_x = np.mean(geometry[:,0])
    centre_y = np.mean(geometry[:,1])

    # shift the traces according to the baz and slow
    shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(slow), baz=float(
        baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

    # Stack the traces (i.e. take mean)
    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

    return lin_stack

@jit(nopython=True, fastmath=True)
def pws_stack_baz_slow(traces, phase_traces, sampling_rate, geometry, distance, slow, baz, degree):
    """
    Function to stack the given traces along a backazimuth and horizontal slownes.

    Param: traces (2D numpy array of floats)
    Description: a 2D numpy array containing the traces that the user wants to stack.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: slow (float)
    Description: horizontal slowness you want to align traces over.

    Param: baz (float)
    Description: backazimuth you want to align traces over.

    Return:
    stack - numpy array of the stacked waveform.
    """
    ntrace = traces.shape[0]

    centre_x = np.mean(geometry[:,0])
    centre_y = np.mean(geometry[:,1])

    # shift the traces according to the baz and slow
    shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(slow), baz=float(
        baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

    shifted_phase_traces = shift_traces(traces=phase_traces, geometry=geometry, abs_slow=float(slow), baz=float(
        baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
    phase_stack = np.absolute(
        np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
    phase_weight_stack = lin_stack * np.power(phase_stack, degree)

    return phase_weight_stack

@jit(nopython=True, fastmath=True)
def get_max_power_loc(tp, sxmin, symin, s_space):
    """
    finds the location of the maximum power value within a given
    slowness space.

    Param: tp (2D array of floats)
    Description: 2D array of values to find the maxima in.

    Param: sxmin (float)
    Description: minimum value on the x axis.

    Param: symin (float)
    Description: minimum value on the y axis.

    Param: s_space (float)
    Description: step interval. Assumes x and y axis spacing is the same.

    return:
        2D array of: [[loc_x,loc_y]]

    """

    peaks = np.empty((1, 2))

    iy, ix = np.where(tp == np.amax(tp))

    slow_x_max = sxmin + (ix[0] * s_space)
    slow_y_max = symin + (iy[0] * s_space)

    # Again, this will be an azimuth plot, not back azimuth
    # So will need to rotate them by 180 to turn into baz.

    peaks[int(0)] = np.array([slow_x_max, slow_y_max])

    return peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_all(traces, phase_traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space, degree):
    """
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using linear, phase weighted stacking and F statistic.


    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smxax (float)
    Description: Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    Param: sxmin (float)
    Description: Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    Param: syxax (float)
    Description: Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    Param: symin (float)
    Description: Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.1.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    ################# Return #################

    - pws_tp: phase weighted stacked power grid.
    - lin_tp: linear stack power grid.
    - f_tp: F-statistic power grid.
    - results_arr: 2D array containing power values for:
        [slow_x, slow_y, power_pws, power_F, power_lin, baz, abs_slow]
    - peaks: 2D array with 3 rows containing the X,Y points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space),0) + 1)
    nsy = int(np.round(((symax - symin) / s_space),0) + 1)

    # make empty array for output.
    results_arr = np.zeros((nsy * nsx, 7))
    lin_tp = np.zeros((nsy, nsx))
    pws_tp = np.zeros((nsy, nsx))
    F_tp = np.zeros((nsy, nsx))


    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            point = int(int(i) + int(slow_xs.shape[0] * j))

            # Call function to shift traces
            shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(abs_slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace


            # linear stack
            power_lin = np.trapz(np.power(lin_stack, 2))

            # phase weighted stack
            shifted_phase_traces = shift_traces(
                traces=phase_traces, geometry=geometry, abs_slow=abs_slow, baz=baz, distance=distance, centre_x=centre_x, centre_y=centre_y, sampling_rate=sampling_rate)

            phase_stack = np.absolute(
                np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            phase_weight_stack = lin_stack * np.power(phase_stack, degree)
            power_pws = np.trapz(np.power(phase_weight_stack, 2))

            # F statistic
            Residuals_Trace_Beam = np.subtract(shifted_traces_lin, lin_stack)
            Residuals_Trace_Beam_Power = np.sum(
                np.power(Residuals_Trace_Beam, 2), axis=0)

            Residuals_Power_Int = np.trapz(Residuals_Trace_Beam_Power)

            F = (shifted_traces_lin.shape[0] - 1) * (
                (shifted_traces_lin.shape[0] * power_lin) / (Residuals_Power_Int))

            lin_tp[i, j] = power_lin
            pws_tp[i, j] = power_pws
            F_tp[i, j] = power_lin * F

            results_arr[point] = np.array(
                [sx, sy, power_pws, power_lin * F, power_lin, baz, abs_slow])


    # lin_tp = np.array(lin_tp)
    # pws_tp = np.array(pws_tp)
    # normalise the array
    # lin_tp /= lin_tp.max()
    # pws_tp /= pws_tp.max()
    # F_tp /= F_tp.max()

    # now find the peak in this:
    peaks = np.empty((3, 2))
    peaks[int(0)] = get_max_power_loc(tp=lin_tp, sxmin=sxmin, symin=symin, s_space=s_space)
    peaks[int(1)] = get_max_power_loc(tp=pws_tp, sxmin=sxmin, symin=symin, s_space=s_space)
    peaks[int(2)] = get_max_power_loc(tp=F_tp, sxmin=sxmin, symin=symin, s_space=s_space)

    results_arr[:, 2] /= results_arr[:, 2].max()
    results_arr[:, 3] /= results_arr[:, 3].max()
    results_arr[:, 4] /= results_arr[:, 4].max()

    return lin_tp, pws_tp, F_tp, results_arr, peaks

@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_Lin(traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space):
    '''
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using linear stack.

    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.


    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smxax (float)
    Description: Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    Param: sxmin (float)
    Description: Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    Param: syxax (float)
    Description: Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    Param: symin (float)
    Description: Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.1.

    ################# Return #################

    - lin_tp: linear stack power grid.
    - results_arr: 2D array containing power values for:
        [slow_x, slow_y, power_pws, power_F, power_lin, baz, abs_slow]
    - peaks: 2D array with 1 row containing the X,Y points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    '''

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space),0) + 1)
    nsy = int(np.round(((symax - symin) / s_space),0) + 1)

    # make empty array for output.
    results_arr = np.zeros((nsy * nsx, 5))
    lin_tp = np.zeros((nsy, nsx))

    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            point = int(int(i) + int(slow_xs.shape[0] * j))

            # Call function to shift traces
            shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(abs_slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace


            # linear stack
            power_lin = np.trapz(np.power(lin_stack, 2))

            lin_tp[i, j] = power_lin

            results_arr[point] = np.array(
                [sx, sy, power_lin, baz, abs_slow])

    # now find the peak in this:
    peaks = np.empty((1, 2))
    peaks[int(0)] = get_max_power_loc(tp=lin_tp, sxmin=sxmin, symin=symin, s_space=s_space)

    results_arr[:, 2] /= results_arr[:, 2].max()

    return lin_tp, results_arr, peaks



@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_PWS(traces, phase_traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space, degree):
    '''
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using phase weighted stacking.

    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smxax (float)
    Description: Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    Param: sxmin (float)
    Description: Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    Param: syxax (float)
    Description: Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    Param: symin (float)
    Description: Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.1.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    ################# Return #################

    - pws_tp: phase weighted stacked power grid.
    - results_arr: 2D array containing power values for:
        [slow_x, slow_y, power_pws, power_F, power_lin, baz, abs_slow]
    - peaks: 2D array with 1 rows containing the X,Y points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    '''

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space),0) + 1)
    nsy = int(np.round(((symax - symin) / s_space),0) + 1)

    # make empty array for output.
    results_arr = np.zeros((nsy * nsx, 5))
    pws_tp = np.zeros((nsy, nsx))

    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            point = int(int(i) + int(slow_xs.shape[0] * j))

            # Call function to shift traces
            shifted_traces_lin = shift_traces(
                traces=traces, geometry=geometry, abs_slow=float(abs_slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

            shifted_phase_traces = shift_traces(
                traces=phase_traces, geometry=geometry, abs_slow=abs_slow, baz=baz, distance=distance, centre_x=centre_x,
                centre_y=centre_y, sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            phase_stack = np.absolute(
                np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            phase_weight_stack = lin_stack * np.power(phase_stack, degree)
            power_pws = np.trapz(np.power(phase_weight_stack, 2))

            pws_tp[i, j] = power_pws

            results_arr[point] = np.array(
                [sx, sy, power_pws, baz, abs_slow])


    # now find the peak in this:
    peaks = np.empty((1, 2))
    peaks[int(0)] = get_max_power_loc(tp=pws_tp, sxmin=sxmin, symin=symin, s_space=s_space)

    results_arr[:, 2] /= results_arr[:, 2].max()

    return pws_tp, results_arr, peaks





@jit(nopython=True, fastmath=True)
def BF_Spherical_Pol_all(traces, phase_traces, sampling_rate, geometry, distance, smin, smax, bazmin, bazmax, s_space, baz_space, degree):
    '''
    Function to search over a range of slowness vectors described in polar coordinates and estimates the
    coherent power. Stacks the traces using linear and phase weighted stacking and applies the F statistic.

    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smax (float)
    Description: Maximum magnitude of slowness.

    Param: smin (float)
    Description: Minimun magnitude of the slowness.

    Param: bazmin (float)
    Description: Minimum backazimuth value to search over.

    Param: bazmax (float)
    Description: Maximum backazimuth value to search over.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.05.

    Param: baz_space (float)
    Description: The backazimuth interval for each step e.g. 0.1.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    ################# Return #################
    5 arrays with: [slow, baz, rel_power] for each slowness and backazimuth combination in the grid for:

    - pws_tp: phase weighted stacked power grid.
    - lin_tp: linear stack power grid.
    - f_tp: F-statistic power grid.
    - results_arr: 2D array containing power values for:
        [slow, baz, power_pws, power_F, power_lin]
    - peaks: 2D array with 3 rows containing the SLOW,BAZ points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    '''

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nslow = int(np.round(((smax - smin) / s_space) + 1))
    nbaz = int(np.round(((bazmax - bazmin) / baz_space) + 1))

    # make empty array for output.
    lin_tp = np.zeros((nslow, nbaz))
    pws_tp = np.zeros((nslow, nbaz))
    F_tp = np.zeros((nslow, nbaz))
    results_arr = np.zeros((nbaz * nslow, 5))

    slows = np.linspace(smin, smax + s_space, nslow)
    bazs = np.linspace(bazmin, bazmax + baz_space, nbaz)

    for i in range(slows.shape[0]):
        for j in range(bazs.shape[0]):

            baz = float(bazs[int(j)])
            slow = float(slows[int(i)])

            # get the slowness and backazimuth of the vector

            # Call function to shift traces
            shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)
            shifted_phase_traces = shift_traces(
                traces=phase_traces, geometry=geometry, abs_slow=slow, baz=baz, distance=distance, centre_x=centre_x, centre_y=centre_y, sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            phase_stack = np.absolute(
                np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            phase_weight_stack = lin_stack * np.power(phase_stack, degree)

            Residuals_Trace_Beam = np.subtract(shifted_traces_lin, lin_stack)
            Residuals_Trace_Beam_Power = np.sum(
                np.power(Residuals_Trace_Beam, 2), axis=0)

            Residuals_Power_Int = np.trapz(Residuals_Trace_Beam_Power)

            power_lin = np.trapz(np.power(lin_stack, 2))
            power_pws = np.trapz(np.power(phase_weight_stack, 2))

            lin_tp[i, j] = power_lin
            pws_tp[i, j] = power_pws
            F = (shifted_traces_lin.shape[0] - 1) * (
                (shifted_traces_lin.shape[0] * power_lin) / (Residuals_Power_Int))

            F_tp[i, j] = power_lin * F
            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array(
                [baz, slow, power_pws, power_lin * F, power_lin])

    # lin_tp = np.array(lin_tp)
    # pws_tp = np.array(pws_tp)
    # normalise the array
    # lin_tp /= lin_tp.max()
    # pws_tp /= pws_tp.max()
    # F_tp /= F_tp.max()

    # now find the peak in this:

    peaks = np.zeros((3, 2))

    iy_lin, ix_lin = np.where(lin_tp == np.amax(lin_tp))
    iy_pws, ix_pws = np.where(pws_tp == np.amax(pws_tp))
    iy_F, ix_F = np.where(F_tp == np.amax(F_tp))

    slow_max_lin = slows[iy_lin[0]]
    baz_max_lin = bazs[ix_lin[0]]

    slow_max_pws = slows[iy_pws[0]]
    baz_max_pws = bazs[ix_pws[0]]

    slow_max_F = slows[iy_F[0]]
    baz_max_F = bazs[ix_F[0]]

    # Numba doesnt like strings...
    # PWS
    # Lin
    # F_stat
    peaks[int(0)] = np.array([baz_max_lin, slow_max_lin])
    peaks[int(1)] = np.array([baz_max_pws, slow_max_pws])
    peaks[int(2)] = np.array([baz_max_F, slow_max_F])

    results_arr[:, 2] /= results_arr[:, 2].max()
    results_arr[:, 3] /= results_arr[:, 3].max()
    results_arr[:, 4] /= results_arr[:, 4].max()

    return lin_tp, pws_tp, F_tp, results_arr, peaks



@jit(nopython=True, fastmath=True)
def BF_Spherical_Pol_Lin(traces, sampling_rate, geometry, distance, smin, smax, bazmin, bazmax, s_space, baz_space):
    '''
    Given a slowness area [sxmin:sxmax,bazmin:bazmax] a beamforming-esque code to estimate the travel times at each station
    using a circular wavefront approximation, align the traces, then instead of linearly stacking, phase weight stacking is performed.
    From the stacked trace the power value is recorded and stored in a square array and written into an xyz file.
    Analysis is on a stream of time series data for a given slowness range, slowness step, frequency band and time window.
    Need to be populated SAC files.

    Before the analysis is performed, the seismograms are cut to a time window between tmin and tmax,
    and the data is bandpass-filtered between frequencies fmin and fmax.

    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smax (float)
    Description: Maximum magnitude of slowness.

    Param: smin (float)
    Description: Minimun magnitude of the slowness.

    Param: bazmin (float)
    Description: Minimum backazimuth value to search over.

    Param: bazmax (float)
    Description: Maximum backazimuth value to search over.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.05.

    Param: baz_space (float)
    Description: The backazimuth interval for each step e.g. 0.1.

    ################# Return #################
    3 arrays with: [slow_x, slow_y, rel_power] for each slowness and backazimuth combination in the grid for:

    - lin_tp: linear stack power grid.
    - results_arr: 2D array containing power values for:
        [slow, baz, power_pws, power_F, power_lin]
    - peaks: 2D array with 3 rows containing the SLOW,BAZ points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    '''

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nslow = int(np.round(((smax - smin) / s_space) + 1))
    nbaz = int(np.round(((bazmax - bazmin) / baz_space) + 1))

    # make empty array for output.
    lin_tp = np.zeros((nslow, nbaz))
    results_arr = np.zeros((nbaz * nslow, 3))

    slows = np.linspace(smin, smax + s_space, nslow)
    bazs = np.linspace(bazmin, bazmax + baz_space, nbaz)

    for i in range(slows.shape[0]):
        for j in range(bazs.shape[0]):

            baz = float(bazs[int(j)])
            slow = float(slows[int(i)])

            # get the slowness and backazimuth of the vector

            # Call function to shift traces
            shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            power_lin = np.trapz(np.power(lin_stack, 2))
            lin_tp[i, j] = power_lin

            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array(
                [baz, slow, power_lin])


    # now find the peak in this:

    peaks = np.zeros((1, 2))

    iy_lin, ix_lin = np.where(lin_tp == np.amax(lin_tp))

    slow_max_lin = slows[iy_lin[0]]
    baz_max_lin = bazs[ix_lin[0]]


    # Numba doesnt like strings...
    # PWS
    # Lin
    # F_stat
    peaks[int(0)] = np.array([baz_max_lin, slow_max_lin])

    results_arr[:, 2] /= results_arr[:, 2].max()

    return lin_tp, results_arr, peaks



@jit(nopython=True, fastmath=True)
def BF_Spherical_Pol_PWS(traces, phase_traces, sampling_rate, geometry, distance, smin, smax, bazmin, bazmax, s_space, baz_space, degree):
    '''
    Given a slowness area [sxmin:sxmax,symin:symax] a beamforming esque code to estimate the travel times at each station
    using a circular wavefront approximation, align the traces, then instead of linearly stacking, phase weight stacking is performed.
    From the stacked trace the power value is recorded and stored in a square array and written into an xyz file.
    Analysis is on a stream of time series data for a given slowness range, slowness step, frequency band and time window.
    Need to be populated SAC files.

    Before the analysis is performed, the seismograms are cut to a time window between tmin and tmax,
    and the data is bandpass-filtered between frequencies fmin and fmax.

    ################# Parameters #################
    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: smax (float)
    Description: Maximum magnitude of slowness.

    Param: smin (float)
    Description: Minimun magnitude of the slowness.

    Param: bazmin (float)
    Description: Minimum backazimuth value to search over.

    Param: bazmax (float)
    Description: Maximum backazimuth value to search over.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.05.

    Param: baz_space (float)
    Description: The backazimuth interval for each step e.g. 0.1.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    ################# Return #################
    3 arrays with: [slow, baz, rel_power] for each slowness and backazimuth combination in the grid for:

    - pws_tp: phase weighted stacked power grid.
    - results_arr: 2D array containing power values for:
        [slow_x, slow_y, power_pws, power_F, power_lin]
    - peaks: 2D array with 3 rows containing the SLOW,BAZ points of the maximum power value for
             phase weighted, linear and F-statistic respectively.
    '''

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = np.mean(geometry[:, 0]), np.mean(
        geometry[:, 1]), np.mean(geometry[:, 2])

    # get number of plen(buff)oints.
    nslow = int(np.round(((smax - smin) / s_space) + 1))
    nbaz = int(np.round(((bazmax - bazmin) / baz_space) + 1))

    # make empty array for output.
    pws_tp = np.zeros((nslow, nbaz))
    results_arr = np.zeros((nbaz * nslow, 3))

    slows = np.linspace(smin, smax + s_space, nslow)
    bazs = np.linspace(bazmin, bazmax + baz_space, nbaz)

    for i in range(slows.shape[0]):
        for j in range(bazs.shape[0]):

            baz = float(bazs[int(j)])
            slow = float(slows[int(i)])

            # get the slowness and backazimuth of the vector

            # Call function to shift traces
            shifted_traces_lin = shift_traces(traces=traces, geometry=geometry, abs_slow=float(slow), baz=float(
                baz), distance=float(distance), centre_x=float(centre_x), centre_y=float(centre_y), sampling_rate=sampling_rate)
            shifted_phase_traces = shift_traces(
                traces=phase_traces, geometry=geometry, abs_slow=slow, baz=baz, distance=distance, centre_x=centre_x, centre_y=centre_y, sampling_rate=sampling_rate)

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            phase_stack = np.absolute(
                np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            phase_weight_stack = lin_stack * np.power(phase_stack, degree)

            power_pws = np.trapz(np.power(phase_weight_stack, 2))

            pws_tp[i, j] = power_pws

            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array(
                [baz, slow, power_pws])

    # now find the peak in this:

    peaks = np.zeros((1, 2))

    iy_pws, ix_pws = np.where(pws_tp == np.amax(pws_tp))


    slow_max_pws = slows[iy_pws[0]]
    baz_max_pws = bazs[ix_pws[0]]

    peaks[int(0)] = np.array([baz_max_pws, slow_max_pws])

    results_arr[:, 2] /= results_arr[:, 2].max()

    return pws_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Noise_Threshold_Relative(traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space):
    """
    Function to calculate the TP plot or power grid given traces and a range of slowness vectors described in
    a Cartesian system (X/Y).

    ################# Parameters #################
    traces: 2D array of floats.
    numpy array of traces
    sampling_rate: int
    Samples per second
    geometry: 2D array of floats.
    2D array of floats with each row containing: [lon, lat, elevation]
    distance: float
    Mean epicentral distance.
    rel_x: float
    The x component of the predicted slowness vector used to align the traces.
    rel_y: float
    The y component of the predicted slowness vector used to align the traces.
    smxax : float
    Maximum magnitude of slowness on x axis, used for creating the slowness grid
    sxmin: float
    Minimun magnitude of the slowness on x axis, used for creating the slowness grid
    syxax : float
    Maximum magnitude of slowness on y axis, used for creating the slowness grid
    symin: float
    Minimun magnitude of the slowness on y axis, used for creating the slowness grid
    s_space: float
    The slowness interval for each step e.g. 0.1.

    ################# Return #################
    lin_tp: 2D array of power values for each slowness vector.
    noise_mean: mean of noise power estimates.
    peaks: peak power location in the grid.
    """







@jit(nopython=True, fastmath=True)
def calculate_locus(P1,P2):
    """
    Function to calculate the locus angle between points P1 and P2.
    A locus is a line that separates two point and is orthogonal to the line P1-P2.

    Param: P1 (1-D numpy array of floats)
    Description: x and y coordinates of point 1.

    Param: P2 (1-D numpy array of floats)
    Description: x and y coordinates of point 2.

    Returns:
        Theta: Angle pointing from P1 to P2 relative to vertical.
        Midpoint: Coordinates of the midpoint between P1 and P2.
        Phi_1: Locus angle (90 degrees from Theta).
        Phi_2: Locus angle (180 degrees from Phi_1).
    """
    P1_x = P1[0]
    P1_y = P1[1]
    P2_x = P2[0]
    P2_y = P2[1]

    del_x = abs(P1_x - P2_x)
    del_y = abs(P1_y - P2_y)

    Theta = np.degrees(np.arctan2(del_x, del_y))

    if Theta < 0:
        Theta += 360

    Midpoint_X = np.mean([P1_x, P2_x])
    Midpoint_Y = np.mean([P1_y, P2_y])
    # Phi will represent the angles from north extending in either direction from
    # the point.

    Phi_1 = Theta + 90
    Phi_2 = Phi_1 + 180

    if Phi_1 > 360:
        Phi_1 -= 360

    if Phi_2 > 360:
        Phi_2 -= 360

    Midpoint = np.array([Midpoint_X, Midpoint_Y])

    print("Theta:", Theta)
    print("Midpoint_Y:", Midpoint_Y)
    print("Midpoint_X:", Midpoint_X)
    print("Phi_1:", Phi_1)
    print("Phi_2:", Phi_2)
    return Theta, Midpoint, Phi_1, Phi_2


@jit(nopython=True, fastmath=True)
def Vespagram_Lin(traces, sampling_rate, geometry, distance, baz, smin, smax, s_space):
    """
    Function to calculate the slowness vespagram of given traces using linear stacking.

    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: baz (float)
    Description: Constant backazimuth value to use.

    Param: smax (float)
    Description: Maximum magnitude of slowness.

    Param: smin (float)
    Description: Minimun magnitude of the slowness.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.05.

    Return:
        ves_lin: 2D numpy array of floats representing how the amplitude of the trace at a slowness
                 varies with time.
    """


    nslow = int(((smax - smin) / s_space) + 1)
    slows = np.linspace(smin, smax+s_space, nslow)

    ves_lin = np.empty((nslow, traces.shape[1]))

    for i in range(slows.shape[0]):

        slow = float(slows[int(i)])

        lin_stack = linear_stack_baz_slow(traces=traces, sampling_rate=sampling_rate, geometry=geometry,
                              distance=distance, slow=slow, baz=baz)

        ves_lin[i] = lin_stack


    return ves_lin




@jit(nopython=True, fastmath=True)
def Vespagram_PWS(traces, phase_traces, sampling_rate, geometry, distance, baz, smin, smax, s_space, degree):
    """
    Function to calculate the slowness vespagram of given traces using phase weighted stacking.

    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: baz (float)
    Description: Constant backazimuth value to use.

    Param: smax (float)
    Description: Maximum magnitude of slowness.

    Param: smin (float)
    Description: Minimun magnitude of the slowness.

    Param: s_space (float)
    Description: The slowness interval for each step e.g. 0.05.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Return:
        ves_pws: 2D numpy array of floats representing how the amplitude of the trace at a slowness
                 varies with time.
    """

    nslow = int(((smax - smin) / s_space) + 1)
    slows = np.linspace(smin, smax + s_space, nslow)

    ves_pws = np.empty((nslow, traces.shape[1]))

    for i in range(slows.shape[0]):

        slow = float(slows[int(i)])

        pws_stack = pws_stack_baz_slow(traces=traces, phase_traces=phase_traces, sampling_rate=sampling_rate,
                                       geometry=geometry, distance=distance, slow=slow, baz=baz, degree=degree)

        ves_pws[i] = pws_stack

    return ves_pws


@jit(nopython=True, fastmath=True)
def Baz_vespagram_Lin(traces, sampling_rate, geometry, distance, slow, bmin, bmax, b_space):
    """
    Function to calculate the backazimuth vespagram of given traces using linear stacking.

    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: slow (float)
    Description: Constant slowness value to use.

    Param: bmax (float)
    Description: Maximum magnitude of backazimuth.

    Param: bmin (float)
    Description: Minimun magnitude of the backazimuth.

    Param: b_space (float)
    Description: The backazimuth interval for each step e.g. 0.05.

    Return:
        ves_lin: 2D numpy array of floats representing how the amplitude of the trace at a slowness
                 varies with time.
    """


    nbaz = int(((bmax - bmin) / b_space) + 1)
    bazs = np.linspace(bmin, bmax+b_space, nbaz)

    ves_lin = np.empty((nbaz, traces.shape[1]))

    for i in range(bazs.shape[0]):

        baz = float(bazs[int(i)])

        lin_stack = linear_stack_baz_slow(traces=traces, sampling_rate=sampling_rate, geometry=geometry,
                              distance=distance, slow=slow, baz=baz)

        ves_lin[i] = lin_stack


    return ves_lin




@jit(nopython=True, fastmath=True)
def Baz_vespagram_PWS(traces, phase_traces, sampling_rate, geometry, distance, slow, bmin, bmax, b_space, degree):
    """
    Function to calculate the backazimuth vespagram of given traces using phase weighted stacking.

    Param: traces (2D numpy array of floats)
    Description: 2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
                 is the number of traces and p is the points in each trace.

    Param: phase_traces (2D numpy array of floats)
    Description: a 2D numpy array containing the instantaneous phase at each time point
                 that the user wants to use in the phase weighted stack. Shape of [n,p]
                 where n is the number of traces and p is the points in each trace.

    Param: sampling_rate (float)
    Description: sampling rate of the data points in s^-1.

    Param: geometry (2D array of floats)
    Description: 2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Param: distance (float)
    Description: Epicentral distance from the event to the centre of the array.

    Param: slow (float)
    Description: Constant slowness value to use.

    Param: bmax (float)
    Description: Maximum magnitude of backazimuth.

    Param: bmin (float)
    Description: Minimun magnitude of the backazimuth.

    Param: b_space (float)
    Description: The backazimuth interval for each step e.g. 0.05.

    Param: degree (float)
    Description: The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Return:
        ves_pws: 2D numpy array of floats representing how the amplitude of the trace at a slowness
                 varies with time.
    """

    nbaz = int(((bmax - bmin) / b_space) + 1)
    bazs = np.linspace(bmin, bmax+b_space, nbaz)

    ves_pws = np.empty((nbaz, traces.shape[1]))

    for i in range(bazs.shape[0]):

        baz = float(bazs[int(i)])

        pws_stack = pws_stack_baz_slow(traces=traces, phase_traces=phase_traces, sampling_rate=sampling_rate,
                                       geometry=geometry, distance=distance, slow=slow, baz=baz, degree=degree)

        ves_pws[i] = pws_stack

    return ves_pws





jit_module(nopython=True, error_model="numpy", fastmath=True)
