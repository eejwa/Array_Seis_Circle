from numba import jit
import numpy as np

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
    for i, sx in enumerate(slow_xs):
        for j, sy in enumerate(slow_ys):

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
            transff[i, j] = np.sum(buff)  # cumtrapz(buff, dx=fstep)[-1]

            point = int(int(j) + int(slow_xs.shape[0] * i))
            ARF_arr[point] = np.array([sx, sy, np.sum(buff)])

    # normalise the array response function
    transff /= transff.max()
    ARF_arr[:, 2] /= ARF_arr[:, 2].max()

    return transff, ARF_arr
