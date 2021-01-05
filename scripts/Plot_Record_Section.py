#!/usr/bin/env python

## code to plot the stations of sac files

import obspy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from array_plotting import plotting

parser = argparse.ArgumentParser(description='Plot record section around target phase using SAC files')
parser.add_argument("-f","--file_path", help="Enter the path to the SAC files (e.g. ./*SAC)", type=str, required=True, action="store")
parser.add_argument("-p","--phase", help="Enter the target phase to plot record section around (e.g. SKS)", type=str, required=True, action="store")
parser.add_argument("-tmin","--time_min", help="Enter the time before the predicted arrival (e.g. 20)", type=float, required=True, action="store")
parser.add_argument("-tmax","--time_max", help="Enter the time after the predicted arrival (e.g. 20)", type=float, required=True, action="store")


args = parser.parse_args()
filepath = args.file_path
phase = args.phase
tmin = args.time_min
tmax = args.time_max

# read in the SAC files
st = obspy.read(filepath)


# plot!
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(121)

p = plotting(ax=ax1)
p.plot_record_section_SAC(st=st, phase=phase, tmin=tmin, tmax=tmax, align=True)

ax2 = fig.add_subplot(122)

p = plotting(ax=ax2)
p.plot_record_section_SAC(st=st, phase=phase, tmin=tmin, tmax=tmax, align=False)

plt.tight_layout()
plt.show()
