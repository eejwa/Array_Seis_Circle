from numba import jit
import numpy as np
from shift_stack import shift_traces, roll_2D, linear_stack_baz_slow
from slow_vec_calcs import get_slow_baz, get_max_power_loc



@jit(nopython=True, fastmath=True)
def BF_Pol_all(
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
    type='circ',
    elevation=False,
    incidence=90
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
                type=type,
                elevation=elevation,
                incidence=incidence
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
                type=type,
                elevation=elevation,
                incidence=incidence
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

            Residuals_Power_Int = np.sum(Residuals_Trace_Beam_Power)

            power_lin = np.sum(lin_stack**2)
            power_pws = np.sum(phase_weight_stack**2)

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
def BF_Pol_Lin(
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
    type='circ',
    elevation=False,
    incidence=90
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

    for i, slow in enumerate(slows):
        for j, baz in enumerate(bazs):

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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            power_lin = np.sum(lin_stack ** 2)

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
def BF_Pol_PWS(
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
    type='circ',
    elevation=False,
    incidence=90
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
                type=type,
                elevation=elevation,
                incidence=incidence
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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack** degree)

            power_pws = np.sum(phase_weight_stack** 2)

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
