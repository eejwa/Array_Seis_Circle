#!/usr/bin/env python

# takes SAC files with stlo, stla headers populated and using inputs
# of min_stations, radius and spacing, breaks up the large array into
# sub arrays

import obspy
import numpy as np
import matplotlib.pyplot as plt
import circ_array as c
import cartopy.crs as ccrs
import cartopy
import argparse
from array_plotting import plotting
from sklearn.neighbors import BallTree

parser = argparse.ArgumentParser(
    description="Break larger array into sub array and plot onto a map"
)

parser.add_argument(
    "-s",
    "--stations",
    help="Minimum number of stations in the sub array (default=10)",
    type=float,
    required=False,
    action="store",
    default=10
)

parser.add_argument(
    "-r",
    "--radius",
    help="The radius of the sub array in degrees (default=2)",
    type=float,
    required=False,
    action="store",
    default=2
)

parser.add_argument(
    "-d",
    "--distance",
    help="The spacing of the sub arrays (default = 2)",
    type=float,
    required=False,
    action="store",
    default=2
)

parser.add_argument(
    "-o",
    "--outfile",
    help="output file path",
    type=str,
    required=False,
    action="store",
    default='./Sub_arrays.txt'
)

parser.add_argument(
    "-f",
    "--filepath",
    help="string describing the data files (default=*SAC)",
    type=str,
    required=False,
    action="store",
    default='./*SAC'
)


args = parser.parse_args()
distance = args.distance
radius = args.radius
min_stat = args.stations
Res_file = args.outfile
filepath = args.filepath

min_dist = np.deg2rad(radius)
spacing =  np.deg2rad(distance)

header = "event centroid_lo centroid_la, n_stations, stations\n"

with open(Res_file, 'w') as w_file:
    w_file.write(header)


# try:
st = obspy.read(filepath)

final_centroids, lats_lons_use, lats_lons_core, stations_use = c.break_sub_arrays(st=st,
                                                                                  min_stat = min_stat,
                                                                                  min_dist = min_dist,
                                                                                  spacing = spacing)


fig = plt.figure(figsize=(10,8), tight_layout=True)

ax1 = fig.add_subplot(111, projection=ccrs.Robinson())


p = plotting(ax1)
p.plot_stations(st)

print(lats_lons_use)


use_tree = BallTree(lats_lons_use, leaf_size=lats_lons_use.shape[0]/2, metric='haversine')
with open(Res_file, 'a') as a_file:

    for i,centroid in enumerate(final_centroids):

        lat_centre = np.around(centroid[0], 2)
        lon_centre = np.around(centroid[1], 2)
        sub_array = use_tree.query_radius(X=[centroid], r=min_dist)[0]
        print(lon_centre, lat_centre)
        print(sub_array)
        stat_string = ",".join(list(stations_use[sub_array]))
        print(f"stations: {len(sub_array)}", f"names: {stations_use[sub_array]}")
        # event_name = os.path.basename(event)
        a_file.write(f"{i} {lon_centre} {lat_centre} {len(sub_array)} {stat_string}\n")
        print('3')

        ax1.scatter(np.degrees(lats_lons_use[:,1][sub_array]), np.degrees(lats_lons_use[:,0][sub_array]),
                    transform=ccrs.Geodetic(), zorder=3)

        ax1.scatter(np.degrees(lon_centre), np.degrees(lat_centre),
                    transform=ccrs.Geodetic(), zorder=3, c='black')

plt.savefig("Sub_array_summary.pdf", type='pdf')
plt.show()
# except:
#     print(f"no data or no dense stations")
