
import numpy as np
from numba import jit
from shift_stack import linear_stack_baz_slow, pws_stack_baz_slow

@jit(nopython=True, fastmath=True)
def Vespagram_Lin(traces, sampling_rate, geometry, distance, baz, smin, smax, s_space, type='circ',
                  elevation=False, incidence=90):
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
            type=type,
            elevation=elevation,
            incidence=incidence
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
    type='circ',
    elevation=False,
    incidence=90
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
            type=type,
            elevation=elevation,
            incidence=incidence
        )

        ves_pws[i] = pws_stack

    return ves_pws


@jit(nopython=True, fastmath=True)
def Baz_vespagram_Lin(
    traces, sampling_rate, geometry, distance, slow, bmin, bmax, b_space, type='circ',
    elevation=False, incidence=90
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
            type=type,
            elevation=elevation,
            incidence=incidence
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
    type='circ',
    elevation=False,
    incidence=90
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
            type=type,
            elevation=elevation,
            incidence=incidence
        )

        ves_pws[i] = pws_stack

    return ves_pws
