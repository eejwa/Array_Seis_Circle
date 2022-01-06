# /usr/bin/env python

from numba import jit, jit_module
import numpy as np
from shift_stack import roll_1D, roll_2D


@jit(nopython=True, fastmath=True)
def coords_lonlat_rad_bearing(lat1, lon1, dist_deg, brng):
    """
    Returns the latitude and longitude of a new cordinate that is the defined distance away and
    at the correct bearing from the starting point.

    Parameters
    ----------
    lat1 : float
        Starting point latitiude.

    lon1 : float
        Starting point longitude.

    dist_deg : float
        Distance from starting point in degrees.

    brng : float
        Angle from north describing the direction where the new coordinate is located.

    Returns
    -------
    lat2 : float
        Longitude of the new cordinate.
    lon2 : float
        Longitude of the new cordinate.
    """

    brng = np.radians(brng)  # convert bearing to radians
    d = np.radians(dist_deg)  # convert degrees to radians
    lat1 = np.radians(lat1)  # Current lat point converted to radians
    lon1 = np.radians(lon1)  # Current long point converted to radians

    lat2 = np.arcsin(
        (np.sin(lat1) * np.cos(d)) + (np.cos(lat1) * np.sin(d) * np.cos(brng))
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    # lon2 = np.where(lon2 > 180, lon2 - 360, lon2)
    # lon2 = np.where(lon2 < -180, lon2 + 360, lon2)

    if lon2 > 180:
        lon2 -= 360
    elif lon2 < -180:
        lon2 += 360
    else:
        pass

    return lat2, lon2


@jit(nopython=True, fastmath=True)
def haversine_deg(lat1, lon1, lat2, lon2):
    """
    Function to calculate the distance in degrees between two points on a sphere.

    Parameters
    ----------
    lat1 : float
        Latitiude of point 1.

    lat1 : float
        Longitiude of point 1.

    lat2 : float
        Latitiude of point 2.

    lon2 : float
        Longitude of point 2.

    Returns
    -------
        d : float
            Distance between the two points in degrees.
    """

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2)) ** 2 + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = np.degrees(c)
    return d


@jit(nopython=True, fastmath=True)
def get_slow_baz(slow_x, slow_y, dir_type):
    """
    Returns the backazimuth and slowness magnitude of a slowness vector given its x and y components.

    Parameters
    ----------
    slow_x : float
        X component of slowness vector.

    slow_y : float
        Y component of slowness vector.

    dir_type : string
        How do you want the direction to be measured, backazimuth (baz) or azimuth (az).

    Returns
    -------
    slow_mag: float
        Magnitude of slowness vector.
    baz : float
        Backazimuth of slowness vector
    azimuth : float
        Azimuth of slowness vector
    """

    slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
    azimuth = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)

    # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
    baz = azimuth % -360 + 180

    # make baz positive if it's negative:
    # baz = np.where(baz < 0, baz + 360, baz)
    # azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
    # baz = np.where(baz > 360, baz - 360, baz)
    # azimuth = np.where(azimuth > 360, azimuth - 360, azimuth)

    if baz < 0:
        baz += 360
    if azimuth < 0:
        azimuth += 360



    if dir_type == "baz":
        return slow_mag, baz
    elif dir_type == "az":
        return slow_mag, azimuth
    else:
        pass

# replace some of this with other functions
@jit(nopython=True)
def ARF_process_f_s_spherical(
    geometry, sxmin, sxmax, symin, symax, sstep, distance, fmin, fmax, fstep, scale
):
    """
    Returns array transfer function as a function of slowness difference.
    This will only work for SAC files.

    Parameters
    ----------
    geometry : 2D numpy array
        Numpy array of [lon, lat,elevation] values for each station.

    s[x/y](min/max) : float.
        The min/max value of the slowness of the wavefront in x/y direction.

    sstep : float
        Slowness interval in x andy direction.

    distance : float
        The distance of the event from the centre of the stations,
        this will be used to estimate the curvature of the wavefront.

    fmin : float
        Minimum frequency in signal.

    fmax : float
        Maximum frequency in signal.

    fstep : float
        Frequency sample distance.

    scale: : bool
        If True, the values will be normalised between 0 and 1.

    Returns
    -------
    transff : 2D numpy array of floats
        Power values in a slowness grid for the array response function.
    ARF_arr: 2D numpy array of floats
        [slow_x,slow_y,power] can be written to an xyz file for GMT.
    """

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
    nsx = int(np.round(((sxmax - sxmin) / s_space) + 1))
    nsy = int((np.round((symax - symin) / s_space) + 1))
    # number of frequencies
    nf = int(np.ceil((fmax + fstep / 10.0 - fmin) / fstep))

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
            abs_slow = np.sqrt(sx ** 2 + sy ** 2)
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

            lat_new = np.arcsin(
                (np.sin(lat1) * np.cos(d)) + (np.cos(lat1) * np.sin(d) * np.cos(brng))
            )
            lon_new = lon1 + np.arctan2(
                np.sin(brng) * np.sin(d) * np.cos(lat1),
                np.cos(d) - np.sin(lat1) * np.sin(lat_new),
            )

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
                a = float(
                    (np.sin(dlat / 2)) ** 2
                    + np.cos(np.radians(lat_new_deg))
                    * np.cos(np.radians(stla))
                    * (np.sin(dlon / 2)) ** 2
                )
                c = float(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
                dist = float(np.degrees(c))

                sta_distances[int(x)] = float(dist)

            # calculate ARF
            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10.0, fstep)):
                _sum = 0j
                for l in range(sta_distances.shape[0]):
                    _sum += np.exp(
                        # Note: the coords need to be in [x,y] i.e. for each coord, 0=x dist and 1=y dist from the centre of the array.
                        # the zero is the amplitude (i.e. the ARF has no amplitude only the distribution)
                        complex(0.0, (sta_distances[int(l)] * abs_slow) * 2 * np.pi * f)
                    )
                buff[int(k)] = abs(_sum) ** 2
            transff[i, j] = np.trapz(buff)  # cumtrapz(buff, dx=fstep)[-1]

            point = int(int(j) + int(slow_xs.shape[0] * i))
            ARF_arr[point] = np.array([sx, sy, np.trapz(buff)])

    # normalise the array response function
    transff /= transff.max()
    ARF_arr[:, 2] /= ARF_arr[:, 2].max()

    return transff, ARF_arr


@jit(nopython=True, fastmath=True)
def calculate_time_shifts(
    geometry, abs_slow, baz, distance, centre_x, centre_y, type="circ"
):
    """
    Calculates the time delay for each station relative to the time the phase
    should arrive at the centre of the array. Will use either a plane or curved
    wavefront approximation.

    Parameters
    ----------

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    abs_slow : float
        Horizontal slowness you want to align traces over.

    baz : float
        Backazimuth you want to align traces over.

    centre_x : float
        Mean longitude.

    centre_y : float
        Mean latitude.

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    Returns
    -------
    times : 1D numpy array of floats
        The arrival time for the phase at
        each station relative to the centre.

    shifts : 1D numpy array of floats
        The time shift to align on a phase at
        each station relative to the centre.
    """

    slow_x = abs_slow * np.sin(np.radians(baz))
    slow_y = abs_slow * np.cos(np.radians(baz))

    lat_new, lon_new = coords_lonlat_rad_bearing(
        lat1=centre_y, lon1=centre_x, dist_deg=distance, brng=baz
    )

    if type == "circ":

        dists = haversine_deg(lat1=lat_new, lon1=lon_new, lat2=geometry[:,1], lon2=geometry[:,0])

        # get the relative distance
        dists_rel = dists - distance

        # get the travel time for this distance
        times = dists_rel * abs_slow

        # the correction will be dt *-1
        shifts = times * -1


    elif type == "plane":

        shifts = ((geometry[:,0] - centre_x) * slow_x) + ((geometry[:,1] - centre_y) * slow_y)

        times = shifts * -1

    else:
        print("not plane or circ")

    return shifts, times


@jit(nopython=True, fastmath=True)
def shift_traces(
    traces,
    geometry,
    abs_slow,
    baz,
    distance,
    centre_x,
    centre_y,
    sampling_rate,
    type="circ",
):
    """
    Shifts the traces using the predicted arrival times for a given backazimuth and slowness.

    Parameters
    ----------
    traces : 2D numpy array of floats
        A 2D numpy array containing the traces that the user wants to stack.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    abs_slow : float
        Horizontal slowness you want to align traces over.

    baz : float
        Backazimuth you want to align traces over.

    centre_x : float
        Mean longitude.

    centre_y : float
        Mean latitude.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    Returns
    -------
        shifted_traces : 2D numpy array of floats
            The input traces shifted by the predicted arrival time
            of a curved wavefront arriving from a backazimuth and
            slowness.
    """


    shifts, times = calculate_time_shifts(
                                          geometry,
                                          abs_slow,
                                          baz,
                                          distance,
                                          centre_x,
                                          centre_y,
                                          type=type
                                          )

    pts_shifts = shifts * sampling_rate

    shifted_traces = roll_2D(traces, pts_shifts)


    return shifted_traces


@jit(nopython=True, fastmath=True)
def linear_stack_baz_slow(traces, sampling_rate, geometry, distance, slow, baz):
    """
    Function to stack the given traces along a backazimuth and horizontal slownes.

    Parameters
    ----------
    traces : 2D numpy array of floats
        A 2D numpy array containing the traces that the user wants to stack.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    Distance : float
        Epicentral distance from the event to the centre of the array.

    slow : float
        Horizontal slowness you want to align traces over.

    baz : float
        Backazimuth you want to align traces over.

    Returns
    -------
    lin_stack : 1D numpy array of floats
        The stacked waveform. The traces are shifted by the backazimuth
        and horizontal slowness before stacking
    """

    ntrace = traces.shape[0]
    centre_x = np.mean(geometry[:, 0])
    centre_y = np.mean(geometry[:, 1])

    # shift the traces according to the baz and slow
    shifted_traces_lin = shift_traces(
        traces=traces,
        geometry=geometry,
        abs_slow=float(slow),
        baz=float(baz),
        distance=float(distance),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
    )

    # Stack the traces (i.e. take mean)
    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

    return lin_stack


@jit(nopython=True, fastmath=True)
def pws_stack_baz_slow(
    traces, phase_traces, sampling_rate, geometry, distance, slow, baz, degree
):
    """
    Function to stack the given traces along a backazimuth and horizontal slownes.

    Parameters
    ----------
    traces : 2D numpy array of floats
        A 2D numpy array containing the traces that the user wants to stack.

    phase_traces : 2D numpy array of floats
        A 2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    slow : float
        Horizontal slowness you want to align traces over.

    baz : float
        Backazimuth you want to align traces over.

    Returns
    -------
    stack : 1D numpy array of floats
        The phase-weighted stacked waveform. The traces are shifted
        using the backazimuth and horizontal slowness before stacking.
    """
    ntrace = traces.shape[0]

    centre_x = np.mean(geometry[:, 0])
    centre_y = np.mean(geometry[:, 1])

    # shift the traces according to the baz and slow
    shifted_traces_lin = shift_traces(
        traces=traces,
        geometry=geometry,
        abs_slow=float(slow),
        baz=float(baz),
        distance=float(distance),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
    )

    # shift the phase traces
    shifted_phase_traces = shift_traces(
        traces=phase_traces,
        geometry=geometry,
        abs_slow=float(slow),
        baz=float(baz),
        distance=float(distance),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
    )

    # get linear and phase stacks
    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
    phase_stack = (
        np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
    )

    # calculate phase weighted stack
    phase_weight_stack = lin_stack * (phase_stack**degree)

    return phase_weight_stack


@jit(nopython=True, fastmath=True)
def get_max_power_loc(tp, sxmin, symin, s_space):
    """
    Finds the location of the maximum power value within a given
    slowness space.

    Parameters
    ----------
    tp : 2D array of floats
        2D array of values to find the maxima in.

    sxmin : float
        Minimum value on the x axis.

    symin : float
        Minimum value on the y axis.

    s_space : float
        Step interval. Assumes x and y axis spacing is the same.

    Returns
    -------
    peaks : 2D numpy array of floats
        2D array of: [[loc_x,loc_y]]
    """

    peaks = np.empty((1, 2))

    iy, ix = np.where(tp == np.amax(tp))

    slow_x_max = sxmin + (ix[0] * s_space)
    slow_y_max = symin + (iy[0] * s_space)

    peaks[int(0)] = np.array([slow_x_max, slow_y_max])

    return peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_all(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    sxmin,
    sxmax,
    symin,
    symax,
    s_space,
    degree,
):
    """
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using linear, phase weighted stacking and F statistic.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    sxmax : float
        Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    sxmin : float
        Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    symax : float
        Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    symin : float
        Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    s_space : float
        The slowness interval for each step e.g. 0.1.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    pws_tp : 2D numpy array of floats.
        Phase weighted stacked power grid.
    lin_tp : 2D numpy array of floats.
        Linear stack power grid.
    f_tp : 2D numpy array of floats.
        F-statistic power grid.
    results_arr : 2D numpy array of floats.
        Contains power values for:
        [slow_x, slow_y, power_pws, power_F,
        power_lin, baz, abs_slow]
    peaks : 2D array of floats.
        Array contains 3 rows describing the X,Y points
        of the maximum power value for phase weighted,
        linear and F-statistic respectively.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
    nsy = int(np.round(((symax - symin) / s_space), 0) + 1)

    # make empty array for output.
    results_arr = np.zeros((nsy * nsx, 7))
    lin_tp = np.zeros((nsy, nsx))
    pws_tp = np.zeros((nsy, nsx))
    F_tp = np.zeros((nsy, nsx))

    # calculate slowness values in the grid
    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    # loop over slowness grid
    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            point = int(int(i) + int(slow_xs.shape[0] * j))

            # Call function to shift traces
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=float(abs_slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            # linear stack
            power_lin = np.trapz(lin_stack**2)

            # phase weighted stack
            shifted_phase_traces = shift_traces(
                traces=phase_traces,
                geometry=geometry,
                abs_slow=abs_slow,
                baz=baz,
                distance=distance,
                centre_x=centre_x,
                centre_y=centre_y,
                sampling_rate=sampling_rate,
            )

            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack**degree)
            power_pws = np.trapz(phase_weight_stack**2)

            # F statistic
            Residuals_Trace_Beam = np.subtract(shifted_traces_lin, lin_stack)
            Residuals_Trace_Beam_Power = np.sum(
                (Residuals_Trace_Beam**2), axis=0
            )

            Residuals_Power_Int = np.trapz(Residuals_Trace_Beam_Power)

            F = (shifted_traces_lin.shape[0] - 1) * (
                (shifted_traces_lin.shape[0] * power_lin) / (Residuals_Power_Int)
            )

            # store values
            lin_tp[i, j] = power_lin
            pws_tp[i, j] = power_pws
            F_tp[i, j] = power_lin * F

            results_arr[point] = np.array(
                [sx, sy, power_pws, power_lin * F, power_lin, baz, abs_slow]
            )

    # now find the peak in this:
    peaks = np.empty((3, 2))
    peaks[int(0)] = get_max_power_loc(
        tp=lin_tp, sxmin=sxmin, symin=symin, s_space=s_space
    )
    peaks[int(1)] = get_max_power_loc(
        tp=pws_tp, sxmin=sxmin, symin=symin, s_space=s_space
    )
    peaks[int(2)] = get_max_power_loc(
        tp=F_tp, sxmin=sxmin, symin=symin, s_space=s_space
    )

    results_arr[:, 2] /= results_arr[:, 2].max()
    results_arr[:, 3] /= results_arr[:, 3].max()
    results_arr[:, 4] /= results_arr[:, 4].max()

    return lin_tp, pws_tp, F_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_Lin(
    traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space
):
    """
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using linear stack.
    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.
    sampling_rate : float
        Sampling rate of the data points in s^-1.
    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]
    distance : float
        Epicentral distance from the event to the centre of the array.
    sxmax : float
        Maximum magnitude of slowness on x axis, used for creating the slowness grid.
    sxmin : float
        Minimun magnitude of the slowness on x axis, used for creating the slowness grid.
    symax : float
        Maximum magnitude of slowness on y axis, used for creating the slowness grid.
    symin : float
        Minimun magnitude of the slowness on y axis, used for creating the slowness grid.
    s_space : float
        The slowness interval for each step e.g. 0.1.
    Returns
    -------
    lin_tp : 2D numpy array of floats.
        Linear stack power grid.
    results_arr : 2D numpy array of floats.
        Contains power values for:
        [slow_x, slow_y, power_lin, baz, abs_slow]
    peaks : 2D array of floats.
        Array contains 3 rows describing the X,Y points
        of the maximum power value for phase weighted,
        linear and F-statistic respectively.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of plen(buff)oints.
    nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
    nsy = int(np.round(((symax - symin) / s_space), 0) + 1)

    # make empty array for output.
    results_arr = np.zeros((nsy * nsx, 5))
    lin_tp = np.zeros((nsy, nsx))

    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    # loop over slowness grid
    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            point = int(int(i) + int(slow_xs.shape[0] * j))

            # Call function to shift traces
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=float(abs_slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            # linear stack
            power_lin = np.trapz(lin_stack**2)

            lin_tp[i, j] = power_lin

            results_arr[point] = np.array([sx, sy, power_lin, baz, abs_slow])

    # now find the peak in this:
    peaks = np.empty((1, 2))
    peaks[int(0)] = get_max_power_loc(
        tp=lin_tp, sxmin=sxmin, symin=symin, s_space=s_space
    )

    results_arr[:, 2] /= results_arr[:, 2].max()

    return lin_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_XY_PWS(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    sxmin,
    sxmax,
    symin,
    symax,
    s_space,
    degree,
):
    """
    Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
    the coherent power. Stacks the traces using phase weighted stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth].

    distance : float
        Epicentral distance from the event to the centre of the array.

    sxmax : float
        Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    sxmin : float
        Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    symax : float
        Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    symin : float
        Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    s_space : float
        The slowness interval for each step e.g. 0.1.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    pws_tp : 2D numpy array of floats.
        Phase weighted stacked power grid.
    results_arr : 2D numpy array of floats.
        Contains power values for:
        [slow_x, slow_y, power_pws, baz, abs_slow]
    peaks : 2D array of floats.
        Array contains 3 rows describing the X,Y points
        of the maximum power value for phase weighted,
        linear and F-statistic respectively.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
    nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
    nsy = int(np.round(((symax - symin) / s_space), 0) + 1)

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
                traces=traces,
                geometry=geometry,
                abs_slow=float(abs_slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )

            shifted_phase_traces = shift_traces(
                traces=phase_traces,
                geometry=geometry,
                abs_slow=abs_slow,
                baz=baz,
                distance=distance,
                centre_x=centre_x,
                centre_y=centre_y,
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack**degree)
            power_pws = np.trapz(phase_weight_stack**2)

            pws_tp[i, j] = power_pws

            results_arr[point] = np.array([sx, sy, power_pws, baz, abs_slow])

    # now find the peak in this:
    peaks = np.empty((1, 2))
    peaks[int(0)] = get_max_power_loc(
        tp=pws_tp, sxmin=sxmin, symin=symin, s_space=s_space
    )

    results_arr[:, 2] /= results_arr[:, 2].max()

    return pws_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_Pol_all(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    smin,
    smax,
    bazmin,
    bazmax,
    s_space,
    baz_space,
    degree,
):
    """
    Function to search over a range of slowness vectors described in polar coordinates and estimates the
    coherent power. Stacks the traces using linear and phase weighted stacking and applies the F statistic.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    smax : float
        Maximum magnitude of slowness.

    smin : float
        Minimun magnitude of the slowness.

    bazmin : float
        Minimum backazimuth value to search over.

    bazmax : float
        Maximum backazimuth value to search over.

    s_space : float
        The slowness interval for each step e.g. 0.05.

    baz_space : float
        The backazimuth interval for each step e.g. 0.1.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    pws_tp : 2D numpy array of floats.
        Phase weighted stacked power grid.
    lin_tp : 2D numpy array of floats.
        Linear stack power grid.
    f_tp : 2D numpy array of floats.
        F-statistic power grid.
    results_arr : 2D numpy array of floats.
        Array contains values for:
        [slow, baz, power_pws, power_F, power_lin]
    peaks : 2D numpy array of floats.
        2D array with 3 rows containing the horizontal_slowness,
        backzimuth points of the maximum power value for
        phase weighted, linear and F-statistic respectively.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
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
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=float(slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )
            shifted_phase_traces = shift_traces(
                traces=phase_traces,
                geometry=geometry,
                abs_slow=slow,
                baz=baz,
                distance=distance,
                centre_x=centre_x,
                centre_y=centre_y,
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack**degree)

            Residuals_Trace_Beam = np.subtract(shifted_traces_lin, lin_stack)
            Residuals_Trace_Beam_Power = np.sum(
                (Residuals_Trace_Beam**2), axis=0
            )

            Residuals_Power_Int = np.trapz(Residuals_Trace_Beam_Power)

            power_lin = np.trapz(lin_stack**2)
            power_pws = np.trapz(phase_weight_stack**2)

            lin_tp[i, j] = power_lin
            pws_tp[i, j] = power_pws
            F = (shifted_traces_lin.shape[0] - 1) * (
                (shifted_traces_lin.shape[0] * power_lin) / (Residuals_Power_Int)
            )

            F_tp[i, j] = power_lin * F
            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array(
                [baz, slow, power_pws, power_lin * F, power_lin]
            )

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
def BF_Spherical_Pol_Lin(
    traces,
    sampling_rate,
    geometry,
    distance,
    smin,
    smax,
    bazmin,
    bazmax,
    s_space,
    baz_space,
):
    """
    Function to search over a range of slowness vectors described in polar coordinates and estimates the
    coherent power. Stacks the traces using linear stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    smax : float
        Maximum magnitude of slowness.

    smin : float
        Minimun magnitude of the slowness.

    bazmin : float
        Minimum backazimuth value to search over.

    bazmax : float
        Maximum backazimuth value to search over.

    s_space : float
        The slowness interval for each step e.g. 0.05.

    baz_space : float
        The backazimuth interval for each step e.g. 0.1.

    Returns
    -------
    lin_tp : 2D numpy array of floats.
        Linear stack power grid.
    results_arr : 2D numpy array of floats.
        Array contains values for:
        [slow, baz, power_lin]
    peaks : 2D numpy array of floats.
        2D array with 3 rows containing the horizontal_slowness,
        backzimuth points of the maximum power value.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
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
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=float(slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            power_lin = np.trapz(lin_stack ** 2)
            lin_tp[i, j] = power_lin

            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array([baz, slow, power_lin])

    # now find the peak in this:

    peaks = np.zeros((1, 2))

    iy_lin, ix_lin = np.where(lin_tp == np.amax(lin_tp))

    slow_max_lin = slows[iy_lin[0]]
    baz_max_lin = bazs[ix_lin[0]]

    peaks[int(0)] = np.array([baz_max_lin, slow_max_lin])

    results_arr[:, 2] /= results_arr[:, 2].max()

    return lin_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Spherical_Pol_PWS(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    smin,
    smax,
    bazmin,
    bazmax,
    s_space,
    baz_space,
    degree,
):
    """
    Function to search over a range of slowness vectors described in polar coordinates and estimates the
    coherent power. Stacks the traces using phase weighted stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    smax : float
        Maximum magnitude of slowness.

    smin : float
        Minimun magnitude of the slowness.

    bazmin : float
        Minimum backazimuth value to search over.

    bazmax : float
        Maximum backazimuth value to search over.

    s_space : float
        The slowness interval for each step e.g. 0.05.

    baz_space : float
        The backazimuth interval for each step e.g. 0.1.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    pws_tp : 2D numpy array of floats.
        Phase weighted stacked power grid.
    results_arr : 2D numpy array of floats.
        Array contains values for:
        [slow, baz, power_pws]
    peaks : 2D numpy array of floats.
        2D array with 3 rows containing the horizontal_slowness,
        backzimuth points of the maximum power value.
    """

    ntrace = traces.shape[0]

    # get geometry in km from a central point. Needs to be in SAC format. :D :D :D
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
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
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=float(slow),
                baz=float(baz),
                distance=float(distance),
                centre_x=float(centre_x),
                centre_y=float(centre_y),
                sampling_rate=sampling_rate,
            )
            shifted_phase_traces = shift_traces(
                traces=phase_traces,
                geometry=geometry,
                abs_slow=slow,
                baz=baz,
                distance=distance,
                centre_x=centre_x,
                centre_y=centre_y,
                sampling_rate=sampling_rate,
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack** degree)

            power_pws = np.trapz(phase_weight_stack** 2)

            pws_tp[i, j] = power_pws

            point = int(int(i) + int(slows.shape[0] * j))
            results_arr[point] = np.array([baz, slow, power_pws])

    # now find the peak in this:

    peaks = np.zeros((1, 2))

    iy_pws, ix_pws = np.where(pws_tp == np.amax(pws_tp))
    slow_max_pws = slows[iy_pws[0]]
    baz_max_pws = bazs[ix_pws[0]]

    peaks[int(0)] = np.array([baz_max_pws, slow_max_pws])

    results_arr[:, 2] /= results_arr[:, 2].max()

    return pws_tp, results_arr, peaks


@jit(nopython=True, fastmath=True)
def BF_Noise_Threshold_Relative_XY(
    traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space
):
    """
    Function to calculate the TP plot or power grid given traces and a
    range of slowness vectors described in a Cartesian system (X/Y).
    Once the power grid has been calculated, a noise estimate is found by scambling
    the traces 1000 times and, in each scamble, stack them then calculate a power
    value.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    sampling_rate : float
         Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    rel_x: float
        The x component of the predicted slowness vector used to align the traces.

    rel_y: float
        The y component of the predicted slowness vector used to align the traces.

    sxmax : float
        Maximum magnitude of slowness on x axis, used for creating the slowness grid.

    sxmin : float
        Minimun magnitude of the slowness on x axis, used for creating the slowness grid.

    symax : float
        Maximum magnitude of slowness on y axis, used for creating the slowness grid.

    symin : float
        Minimun magnitude of the slowness on y axis, used for creating the slowness grid.

    s_space : float
        The slowness interval for each step e.g. 0.1.

    Returns
    -------
    lin_tp : 2D array of floats
        Power values for each slowness vector.
    noise_mean : float
        Mean of noise power estimates.
    peaks : 2D numpy array of floats
        XY point of the peak power location in the grid.
    """

    # number of traces
    ntrace = traces.shape[0]

    # get mean station location from geometry
    centre_x, centre_y, centre_z = (
        np.mean(geometry[:, 0]),
        np.mean(geometry[:, 1]),
        np.mean(geometry[:, 2]),
    )

    # get number of points.
    nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
    nsy = int(np.round(((symax - symin) / s_space), 0) + 1)

    # make empty array for output.
    lin_tp = np.zeros((nsy, nsx))
    noise_arr = np.zeros((nsy, nsx))

    # slowness grid points
    slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
    slow_ys = np.linspace(symin, symax + s_space, nsy)

    #  loop over slowness vectors
    for i in range(slow_ys.shape[0]):
        for j in range(slow_xs.shape[0]):

            sx = float(slow_xs[int(j)])
            sy = float(slow_ys[int(i)])

            # get the slowness and backazimuth of the vector
            abs_slow, baz = get_slow_baz(sx, sy, "az")

            # Call function to shift traces
            shifted_traces_lin = shift_traces(
                traces=traces,
                geometry=geometry,
                abs_slow=np.array(abs_slow),
                baz=np.array(baz),
                distance=distance,
                centre_x=centre_x,
                centre_y=centre_y,
                sampling_rate=sampling_rate,
            )

            # stack, get power and store in array
            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            power_lin = np.trapz(lin_stack**2)
            lin_tp[i, j] = power_lin

    # initialise peak array
    peaks = np.zeros((1, 2))

    # find highest power value
    iy_lin, ix_lin = np.where(lin_tp == np.amax(lin_tp))

    slow_x_max_lin = sxmin + (ix_lin[0] * s_space)
    slow_y_max_lin = symin + (iy_lin[0] * s_space)

    # find the maximum location
    slow_x_max_lin_abs = slow_x_max_lin  # + rel_x
    slow_y_max_lin_abs = slow_y_max_lin  # + rel_y

    peaks[int(0)] = np.array([slow_x_max_lin_abs, slow_y_max_lin_abs])
    max_peak = peaks[int(0)]
    # get the slowness and baz for the max points
    slow_max_lin, baz_max_lin = get_slow_baz(
        slow_x_max_lin_abs, slow_y_max_lin_abs, "az"
    )

    #### Noise power estimate ####
    # get best_lin_traces plot using baz/slow of peak
    shifted_traces_t0 = shift_traces(
        traces=traces,
        geometry=geometry,
        abs_slow=float(slow_max_lin),
        baz=float(baz_max_lin),
        distance=float(distance),
        centre_x=float(centre_x),
        centre_y=float(centre_y),
        sampling_rate=sampling_rate,
    )

    #  Now need to get values for noise time shift
    # idea is to distort this best linear stack
    # by scrambling the traces randomly

    # T is the max time is can be shifted by
    #  I have made it the length of the trace/2
    T = (traces[0].shape[0] / sampling_rate) / 2.0
    # r values store the fraction of T used to scramble the trace.
    # there is one r value per trace
    # array initialised here
    r_values = np.zeros(shifted_traces_t0.shape[0])

    # initialise array to store noise values.
    noise_powers = np.zeros(1000)

    # scamble 1000 times
    for t in range(1000):
        # get random r values
        for r in range(r_values.shape[0]):
            r_val = np.random.uniform(float(-1), float(1))
            r_values[r] = r_val

        # calculate times as a fraction of T
        added_times = T * r_values

        # now apply these random time shifts to the aligned traces
        noise_traces = np.zeros(shifted_traces_t0.shape)

        # noise_stack = np.zeros(lin_stack.shape)
        for z in range(noise_traces.shape[0]):
            pts_shift_noise = int(added_times[int(z)] * sampling_rate)
            shift_trace_noise = np.roll(shifted_traces_t0[int(z)], pts_shift_noise)
            noise_traces[int(z)] = shift_trace_noise

        noise_stack = np.sum(noise_traces, axis=0) / ntrace

        noise_p = np.trapz(noise_stack**2)
        noise_powers[int(t)] = noise_p

    # take mean of all 1000 values
    noise_mean = np.median(noise_powers)
    noise_arr = np.full_like(lin_tp, noise_p)

    return lin_tp, noise_mean, peaks


@jit(nopython=True, fastmath=True)
def calculate_locus(P1, P2):
    """
    Function to calculate the locus angle between points P1 and P2.
    A locus is a line that separates two point and is orthogonal to the line P1-P2.

    Parameters
    ----------
    P1 : 1-D numpy array of floats
        X and y coordinates of point 1.

    P2 : 1-D numpy array of floats
        X and y coordinates of point 2.

    Returns
    -------
    Theta : float
        Angle pointing from P1 to P2 relative to vertical.
    Midpoint : 1D numpy array of floats
        Coordinates of the midpoint between P1 and P2.
    Phi_1 : float
        Locus angle (90 degrees from Theta).
    Phi_2 : float
        Locus angle (180 degrees from Phi_1).
    """
    P1_x = P1[0]
    P1_y = P1[1]
    P2_x = P2[0]
    P2_y = P2[1]

    del_x = P1_x - P2_x
    del_y = P1_y - P2_y
    print(del_x,del_y)

    Theta = np.degrees(np.arctan2(del_x, del_y))

    if Theta < 0:
        Theta += 360

    Midpoint_X = (P1_x + P2_x)/2
    Midpoint_Y = (P1_y + P2_y)/2
    # Phi will represent the angles from north extending in either direction from
    # the point.

    Phi_1 = Theta + 90
    Phi_2 = Phi_1 + 180

    if Phi_1 > 360:
        Phi_1 -= 360

    if Phi_2 > 360:
        Phi_2 -= 360

    Midpoint = np.array([Midpoint_X, Midpoint_Y])

    # print("Theta:", Theta)
    # print("Midpoint_Y:", Midpoint_Y)
    # print("Midpoint_X:", Midpoint_X)
    # print("Phi_1:", Phi_1)
    # print("Phi_2:", Phi_2)
    return Theta, Midpoint, Phi_1, Phi_2


@jit(nopython=True, fastmath=True)
def Vespagram_Lin(traces, sampling_rate, geometry, distance, baz, smin, smax, s_space):
    """
    Function to calculate the slowness vespagram of given traces using linear stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    baz : float
        Constant backazimuth value to use.

    smax : float
        Maximum magnitude of slowness.

    smin : float
        Minimun magnitude of the slowness.

    s_space : float
        The slowness interval for each step e.g. 0.05.

    Returns
    -------
    ves_lin : 2D numpy array of floats
        Vespagram showing how the amplitude of the trace
        stacked along a horizontal slowness varies with time.
    """

    nslow = int(((smax - smin) / s_space) + 1)
    slows = np.linspace(smin, smax + s_space, nslow)

    ves_lin = np.empty((nslow, traces.shape[1]))

    for i in range(slows.shape[0]):

        slow = float(slows[int(i)])

        lin_stack = linear_stack_baz_slow(
            traces=traces,
            sampling_rate=sampling_rate,
            geometry=geometry,
            distance=distance,
            slow=slow,
            baz=baz,
        )

        ves_lin[i] = lin_stack

    return ves_lin


@jit(nopython=True, fastmath=True)
def Vespagram_PWS(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    baz,
    smin,
    smax,
    s_space,
    degree,
):
    """
    Function to calculate the slowness vespagram of given traces using phase weighted stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    baz : float
        Constant backazimuth value to use.

    smax : float
        Maximum magnitude of slowness.

    smin : float
        Minimun magnitude of the slowness.

    s_space : float
        The slowness interval for each step e.g. 0.05.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    ves_pws : 2D numpy array of floats
        Vespagram showing how the amplitude of the trace
        stacked along a horizontal slowness varies with time.
        Traces are stacked using phase weighted stacking.
    """

    nslow = int(((smax - smin) / s_space) + 1)
    slows = np.linspace(smin, smax + s_space, nslow)

    ves_pws = np.empty((nslow, traces.shape[1]))

    for i in range(slows.shape[0]):

        slow = float(slows[int(i)])

        pws_stack = pws_stack_baz_slow(
            traces=traces,
            phase_traces=phase_traces,
            sampling_rate=sampling_rate,
            geometry=geometry,
            distance=distance,
            slow=slow,
            baz=baz,
            degree=degree,
        )

        ves_pws[i] = pws_stack

    return ves_pws


@jit(nopython=True, fastmath=True)
def Baz_vespagram_Lin(
    traces, sampling_rate, geometry, distance, slow, bmin, bmax, b_space
):
    """
    Function to calculate the backazimuth vespagram of given traces using linear stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    slow : float
        Constant slowness value to use.

    bmax : float
        Maximum backazimuth.

    bmin : float
        Minimum backazimuth.

    b_space : float
        The backazimuth interval for each step e.g. 0.05.

    Returns
    -------
    ves_lin : 2D numpy array of floats
        Vespagram showing how the amplitude of the trace
        stacked using a particular backazimuth varies with time.
    """

    nbaz = int(((bmax - bmin) / b_space) + 1)
    bazs = np.linspace(bmin, bmax + b_space, nbaz)

    ves_lin = np.empty((nbaz, traces.shape[1]))

    for i in range(bazs.shape[0]):

        baz = float(bazs[int(i)])

        lin_stack = linear_stack_baz_slow(
            traces=traces,
            sampling_rate=sampling_rate,
            geometry=geometry,
            distance=distance,
            slow=slow,
            baz=baz,
        )

        ves_lin[i] = lin_stack

    return ves_lin


@jit(nopython=True, fastmath=True)
def Baz_vespagram_PWS(
    traces,
    phase_traces,
    sampling_rate,
    geometry,
    distance,
    slow,
    bmin,
    bmax,
    b_space,
    degree,
):
    """
    Function to calculate the backazimuth vespagram of given traces using phase weighted stacking.

    Parameters
    ----------
    traces : 2D numpy array of floats
        2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
        is the number of traces and p is the points in each trace.

    phase_traces : 2D numpy array of floats
        2D numpy array containing the instantaneous phase at each time point
        that the user wants to use in the phase weighted stack. Shape of [n,p]
        where n is the number of traces and p is the points in each trace.

    sampling_rate : float
        Sampling rate of the data points in s^-1.

    geometry : 2D array of floats
        2D array describing the lon lat and elevation of the stations [lon,lat,depth]

    distance : float
        Epicentral distance from the event to the centre of the array.

    slow : float
        Constant slowness value to use.

    bmax : float
        Maximum backazimuth.

    bmin : float
        Minimum backazimuth.

    b_space : float
        The backazimuth interval for each step e.g. 0.05.

    degree : float
        The degree for the phase weighted stacking to reduce incoherent arrivals by.

    Returns
    -------
    ves_pws : 2D numpy array of floats
        Vespagram showing how the amplitude of the trace
        stacked using different backazimuths varies with time.
        Traces are stacked using phase weighted stacking.
    """

    nbaz = int(((bmax - bmin) / b_space) + 1)
    bazs = np.linspace(bmin, bmax + b_space, nbaz)

    ves_pws = np.empty((nbaz, traces.shape[1]))

    for i in range(bazs.shape[0]):

        baz = float(bazs[int(i)])

        pws_stack = pws_stack_baz_slow(
            traces=traces,
            phase_traces=phase_traces,
            sampling_rate=sampling_rate,
            geometry=geometry,
            distance=distance,
            slow=slow,
            baz=baz,
            degree=degree,
        )

        ves_pws[i] = pws_stack

    return ves_pws


jit_module(nopython=True, error_model="numpy", fastmath=True)
