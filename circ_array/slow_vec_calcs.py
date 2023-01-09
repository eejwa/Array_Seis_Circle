from numba import jit
import numpy as np


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

    return Theta, Midpoint, Phi_1, Phi_2


# @jit(nopython=True, fastmath=True)
def get_slow_baz_array(slow_x, slow_y, dir_type):
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
    baz = np.where(baz < 0, baz + 360, baz)
    azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
    baz = np.where(baz > 360, baz - 360, baz)
    azimuth = np.where(azimuth > 360, azimuth - 360, azimuth)

    # if baz < 0:
    #     baz += 360
    # if azimuth < 0:
    #     azimuth += 360



    if dir_type == "baz":
        return slow_mag, baz
    elif dir_type == "az":
        return slow_mag, azimuth
    else:
        pass

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
