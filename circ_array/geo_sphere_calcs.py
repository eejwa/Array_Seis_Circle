from numba import jit
import numpy as np
from obspy.taup import TauPyModel
import os

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


def deg_km_az_baz(lat1, lon1, lat2, lon2):
    """
    Function to return the ditances in degrees and km over a spherical Earth
    with the backazimuth and azimuth. Distances calculated using the haversine
    formula.

    Parameters
    ----------
    lat(1/2) : float
        Latitude of point (1/2)

    lon(1/2) : float
        Longitude of point (1/2)

    Returns
    -------
    dist_deg : float
        Distance between points in degrees.
    dist_km :
        Distance between points in km.
    az : float
        Azimuth at location 1 pointing to point 2.
    baz : float
        Backzimuth at location 2 pointing to point 1.
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


def relocate_event_baz_slow(evla, evlo, evdp, stla, stlo, baz, slow, phase, mod='prem'):
    """
    Given event location, mean station location and slowness vector
    (baz and slow), relocate the event so the ray arrives with the
    slowness and backazimuth.

    Paramters
    ---------
    evla : float
        Event latitude.
    evlo : float
        Event longitude.
    evdp : float
        Event depth.
    stla : float
        Station latitude.
    stlo : float
        Station longitude.
    baz : float
        Backazimuth of slowness vector.
    slow : float
        Horizontal slowness of slowness vector.
    phase : string
        Target phase (e.g. SKS).
    mod : string
        1D velocity model to use (default is PREM).

    Returns
    -------
    new_evla : float
        Relocated event latitude.
    new_evlo : float
        Relocated event longitude.
    """

    model = TauPyModel(model=mod)

    dist_deg = haversine_deg(lat1=evla, lon1=evlo, lat2=stla, lon2=stlo)

    # define distances to search over
    dist_min=dist_deg-30
    dist_max=dist_deg+30
    dist_search = np.linspace(dist_min, dist_max, 1000)

    # set count so it know if it has found a suitable distance
    count=0
    diff_slows = np.ones(dist_search.shape)
    # if the difference just keeps increasing
    # stop after 20 increases
    early_stop_count = 0
    for i,test_distance in enumerate(dist_search):

        try:
             ## work out slowness and compare to the observed slowness
            tap_out_test = model.get_travel_times(source_depth_in_km=float(evdp),
                                                  distance_in_degree=float(test_distance),
                                                  receiver_depth_in_km=0.0,
                                                  phase_list=[phase])

            abs_slow_test = tap_out_test[0].ray_param_sec_degree
            diff_slow = abs(abs_slow_test - slow)

            ## work out slowness and compare to the observed slowness
            diff_slows[i] = diff_slow

            if diff_slow > diff_slows[i-1]:
                early_stop_count +=1
            else:
                early_stop_count = 0

            if early_stop_count > 20:
                print('increasing risidual for more than 20 iterations, breaking loop')
                break


        except:
            pass

    min = np.amin(np.array(diff_slows))
    loc = np.where(np.array(diff_slows) == min)[0][0]
    distance_at_slowness = dist_search[loc]

    new_evla, new_evlo = coords_lonlat_rad_bearing(lat1 = stla,
                                                   lon1 = stlo,
                                                   dist_deg = distance_at_slowness,
                                                   brng = baz)

    return new_evla, new_evlo


def predict_pierce_points(evla, evlo, evdp, stla, stlo, phase, target_depth, mod='prem'):
    """
    Given station and event locations, return the pierce points at a particular
    depth for source or receiver side locations.

    Parameters
    ----------
    evla : float
        Event latitude.
    evlo : float
        Event longitude.
    evdp : float
        Event depth.
    stla : float
        Station latitude.
    stlo : float
        Station longitude.
    phase : string
        Target phase
    target_depth : float
        Depth to calculate pierce points.
    mod : string
        1D velocity model to use (default is PREM).

    Returns
    -------
    r_pierce_la : float
        Receiver pierce point latitude.
    r_pierce_lo : float
        Receiver pierce point longitude.

    s_pierce_la : float
        Source pierce point latitude.
    s_pierce_lo : float
        Source pierce point longitude.
    """

    # I dont like the obspy taup pierce thing so will use
    # the java script through python.
    # This will assume you have taup installed:
    # https://github.com/crotwell/TauP/

    # print(f"taup_pierce -mod {mod} -h {evdp} -sta {stla} {stlo} -evt {evla} {evlo} -ph {phase} --pierce {target_depth} --nodiscon > ./temp.txt")
    os.system(f"taup_pierce -mod {mod} -h {evdp} -sta {stla} {stlo} -evt {evla} {evlo} -ph {phase} --pierce {target_depth} --nodiscon > ./temp.txt")

    # check number of lines
    with open("./temp.txt", 'r') as temp_file:
        lines_test = temp_file.readlines()
        number_of_lines_test = len(lines_test)

    with open("./temp.txt", 'r') as temp_file:
        lines = temp_file.readlines()
        number_of_lines = len(lines)

        if number_of_lines == 2:
            print(f"Only pierces depth {target_depth} once.")
            print(f"Writing this one line to the file.")
            source_line = lines[-1]
            receiver_line = lines[-1]

        elif number_of_lines == 3:
            source_line = lines[1]
            receiver_line = lines[-1]

        elif number_of_lines > 3:
            print(f"Phase {phase} pierces depth {target_depth} more than twice.")
            print(f"Writing pierce point closest to source/receiver")
            source_line = lines[1]
            receiver_line = lines[-1]


    if number_of_lines != 0:
        s_dist, s_pierce_depth, s_time, s_pierce_la, s_pierce_lo = source_line.split()
        r_dist, r_pierce_depth, r_time, r_pierce_la, r_pierce_lo = receiver_line.split()
    else:
        print('Neither the phase nor ScS can predict this arrival, not continuing')
        s_pierce_la = 'nan'
        s_pierce_lo = 'nan'
        r_pierce_la = 'nan'
        r_pierce_lo = 'nan'
    # os.remove("./temp.txt")
    return s_pierce_la, s_pierce_lo, r_pierce_la, r_pierce_lo
