#!/usr/bin/env python

## code to plot the stations of sac files

import obspy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from array_plotting import plotting

parser = argparse.ArgumentParser(description='Plot stations in SAC files')
parser.add_argument("-f","--file_path", help="Enter the path to the SAC files (e.g. ./*SAC)", type=str, required=True, action="store")
args = parser.parse_args()
filepath = args.file_path

# read in the SAC files
st = obspy.read(filepath)



fig = plt.figure(figsize=(6,6))

# need to give a projection for the cartopy package
# to work. See a list here:
# https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

p = plotting(ax=ax)
p.plot_stations(st)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0, color='gray', alpha=0.5, linestyle='--')

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection=ccrs.Robinson())

p = plotting(ax=ax)
p.plot_paths(st)


plt.show()
