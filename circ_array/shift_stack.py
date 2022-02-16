# /usr/bin/env python

from numba import jit
import numpy as np
from geo_sphere_calcs import coords_lonlat_rad_bearing, haversine_deg


@jit(nopython=True)
def roll_1D(x, p):
    """
    Function to shift traces stored in a 1D array (x)
    by the number of points stored in 1D array (p).
    optimised in Numba.

    Parameters
    ----------
    x : 1D array of floats
        array to be shifted

    p : int
        points to shift array by.

    Returns
    -------
    x : 1D array of floats
        shifted array by points p.

    """

    p = p*-1
    x = np.append(x[p:], x[:p])
    return x

@jit(nopython=True)
def roll_2D(array, shifts):
    """
    Function to shift traces stored in a 2D array (x)
    by the number of points stored in 1D array (p).
    optimised in Numba.

    Parameters
    ----------
    array : 2D array of floats
        Traces/time series to be shifted.

    shifts : 1D array of floats
        points to shift the respective time series by.

    Returns
    -------
    array_new : 2D array of floats
        2D array of shifted time series.

    """

    n = array.shape[0]
    array_new = np.copy(array)
    for i in range(n):
        array_new[int(i)] = roll_1D(array_new[int(i)],int(shifts[int(i)]))
        # array_new[i] = np.roll(array[i],int(shifts[i]))


    return array_new

@jit(nopython=True)
def my_sum(array):
    """
    Function to sum a 1D array.

    Parameters
    ----------
    array : 1D array of floats or integers
        Array to be summed

    Returns
    -------
    total : float
        Sum of values in array.

    """

    total = 0

    for i in range(len(array)):
        total += array[i]

    return total



@jit(nopython=True)
def stack_2D(t):
    """
    Takes the mean across a 2D array (t) as if 'stacking'

    Parameters
    ----------
    t : 2D array of floats
        stacks the time series stored row wise.

    Returns
    -------
    meant : 1D array of floats
        The stacked/averaged time series.
    """

    nt = t.shape[0]
    lt = t.shape[1]
    sumt = np.zeros(lt)
    for ti in t:
        sumt += ti
    meant = sumt / nt
    return meant


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
def calculate_time_shifts_elevation(
    incidence, geometry, abs_slow, baz, distance, centre_x, centre_y, type="circ"
):
    """
    Calculates the time delay for each station relative to the time the phase
    should arrive at the centre of the array. Will use either a plane or curved
    wavefront approximation. This function also adds the relative time shifts
    due to elevation differences of the stations.

    Parameters
    ----------
    incidence : float
        The incidence angle of the phase at the centre of the array.

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

    # Now need to add the effect of station elevation on times and shifts
    # multiply the relative elevations by the cosine of the incidence
    # of the phase and the 1-D velocity of PREM (approx 4.5).
    elevation_times = (geometry[:,2] - np.mean(geometry[:,2])) * np.cos(incidence) * 4.5

    times = times + elevation_times
    shifts = shifts + (elevation_times * -1)

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
    elevation=False,
    incidence=90,
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

    elevation : bool
        If True, elevation corrections will be added. If False, no elevation
        corrections will be accounted for. Default is False.

    incidence : float
        Not used unless elevation is True. Give incidence angle from vertical
        at the centre of the array to calculate elevation corrections.
        Default is 90.

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    Returns
    -------
        shifted_traces : 2D numpy array of floats
            The input traces shifted by the predicted arrival time
            of a curved wavefront arriving from a backazimuth and
            slowness.
    """

    if elevation == False:
        shifts, times = calculate_time_shifts(
                                              geometry,
                                              abs_slow,
                                              baz,
                                              distance,
                                              centre_x,
                                              centre_y,
                                              type=type,
                                              )
    elif elevation == True:
        shifts, times = calculate_time_shifts_elevation(
                                              incidence,
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
def linear_stack_baz_slow(traces, sampling_rate, geometry, distance, slow, baz, type='circ',
    elevation=False, incidence=90):
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

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    elevation : bool
        If True, elevation corrections will be added. If False, no elevation
        corrections will be accounted for. Default is False.

    incidence : float
        Not used unless elevation is True. Give incidence angle from vertical
        at the centre of the array to calculate elevation corrections. Default is 90.

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
        type=type,
        elevation=elevation,
        incidence=incidence
    )

    # Stack the traces (i.e. take mean)
    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

    return lin_stack


@jit(nopython=True, fastmath=True)
def pws_stack_baz_slow(
    traces, phase_traces, sampling_rate, geometry, distance, slow, baz, degree, type='circ',
    elevation=False, incidence=90
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

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    elevation : bool
        If True, elevation corrections will be added. If False, no elevation
        corrections will be accounted for. Default is False.

    incidence : float
        Not used unless elevation is True. Give incidence angle from vertical
        at the centre of the array to calculate elevation corrections. Default is 90.

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
        type=type,
        elevation=elevation,
        incidence=incidence
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
        type=type,
        elevation=elevation,
        incidence=incidence
    )

    # get linear and phase stacks
    lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
    phase_stack = (
        np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
    )

    # calculate phase weighted stack
    phase_weight_stack = lin_stack * (phase_stack**degree)

    return phase_weight_stack
