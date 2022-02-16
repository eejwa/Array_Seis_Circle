
from numba import jit
import numpy as np
from shift_stack import shift_traces, roll_2D
from slow_vec_calcs import get_slow_baz, get_max_power_loc


@jit(nopython=True, fastmath=True)
def BF_XY_all(
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
    type='circ',
    elevation=False,
    incidence=90,
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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            # linear stack
            power_lin = np.sum(lin_stack**2)

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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack**degree)
            power_pws = np.sum(phase_weight_stack**2)

            # F statistic
            Residuals_Trace_Beam = np.subtract(shifted_traces_lin, lin_stack)
            Residuals_Trace_Beam_Power = np.sum(
                (Residuals_Trace_Beam**2), axis=0
            )

            Residuals_Power_Int = np.sum(Residuals_Trace_Beam_Power)

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
def BF_XY_Lin(
    traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space, type='circ',
    elevation=False, incidence=90
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
    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront. default
        is 'circ'.
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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            # linear stack
            power_lin = np.sum(lin_stack**2)

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
def BF_XY_PWS(
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
    type='circ',
    elevation=False,
    incidence=90
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
                type=type,
                elevation=elevation,
                incidence=incidence
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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace

            phase_stack = (
                np.absolute(np.sum(np.exp(shifted_phase_traces * 1j), axis=0)) / ntrace
            )
            phase_weight_stack = lin_stack * (phase_stack**degree)
            power_pws = np.sum(phase_weight_stack**2)

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
def BF_Noise_Threshold_Relative_XY(
    traces, sampling_rate, geometry, distance, sxmin, sxmax, symin, symax, s_space, type='circ',
    elevation=False, incidence=90
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

    type : string
        Will calculate either using a curved (circ) or plane (plane) wavefront.

    elevation : bool
        If True, elevation corrections will be added. If False, no elevation
        corrections will be accounted for. Default is False.

    incidence : float
        Not used unless elevation is True. Give incidence angle from vertical
        at the centre of the array to calculate elevation corrections. Default is
        None.

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
    for i, sy in enumerate(slow_ys):
        for j, sx in enumerate(slow_xs):

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
                type=type,
                elevation=elevation,
                incidence=incidence
            )

            # stack, get power and store in array
            lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
            power_lin = np.sum(lin_stack**2)
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
        type=type,
        elevation=elevation,
        incidence=incidence
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
    # r_values = np.zeros(shifted_traces_t0.shape[0])

    # initialise array to store noise values.
    noise_powers = np.zeros(1000)

    # scamble 1000 times
    for t in range(1000):
        # get random r values
        r_values = np.random.uniform(float(-1), float(1), shifted_traces_t0.shape[0])

        # for r in range(r_values.shape[0]):
        #     r_values[r] = r_val

        # calculate times as a fraction of T
        added_times = T * r_values

        # now apply these random time shifts to the aligned traces
        noise_traces = np.zeros(shifted_traces_t0.shape)

        pts_shift_noise = added_times * sampling_rate

        noise_traces = roll_2D(shifted_traces_t0, pts_shift_noise)

        noise_stack = np.sum(noise_traces, axis=0) / ntrace

        noise_p = np.sum(noise_stack**2)
        noise_powers[int(t)] = noise_p

    # take mean of all 1000 values
    noise_mean = np.median(noise_powers)
    noise_arr = np.full_like(lin_tp, noise_p)

    return lin_tp, noise_mean, peaks
