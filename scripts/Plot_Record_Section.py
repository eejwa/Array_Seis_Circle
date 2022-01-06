#!/usr/bin/env python

## code to plot the stations of sac files

import obspy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from array_plotting import plotting

parser = argparse.ArgumentParser(
    description="Plot record section around target phase using SAC files"
)
parser.add_argument(
    "-f",
    "--file_path",
    help="Enter the path to the SAC files (e.g. ./*SAC)",
    type=str,
    required=True,
    action="store",
)
parser.add_argument(
    "-p",
    "--phase",
    help="Enter the target phase to plot record section around (e.g. SKS), default is None",
    type=str,
    required=False,
    action="store",
    default=None
)
parser.add_argument(
    "-tmin",
    "--time_min",
    help="Enter the time before the predicted arrival (e.g. 20)",
    type=float,
    required=True,
    action="store",
)
parser.add_argument(
    "-tmax",
    "--time_max",
    help="Enter the time after the predicted arrival (e.g. 20)",
    type=float,
    required=True,
    action="store",
)

parser.add_argument(
    "-filter",
    "--filter_question",
    help="Filter traces?",
    required=False,
    action="store_true",
    default=False
)
parser.add_argument(
    "-fmin",
    "--freq_min",
    help="Minimum frequency (default = 0.05)",
    type=float,
    required=False,
    action="store",
    default=0.05
)
parser.add_argument(
    "-fmax",
    "--freq_max",
    help="Maximum frequency (default = 1.0)",
    type=float,
    required=False,
    action="store",
    default=0.5
)


args = parser.parse_args()
filepath = args.file_path
phase = args.phase
tmin = args.time_min
tmax = args.time_max
Filter=args.filter_question
fmin = args.freq_min
fmax = args.freq_max


# read in the SAC files
st = obspy.read(filepath)

if Filter:
    st = st.filter(type='bandpass', freqmin=fmin, freqmax=fmax)
else:
    pass

print(st)

#  plot!
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(121)

p = plotting(ax=ax1)
p.plot_record_section_SAC(st=st, phase=phase, tmin=tmin, tmax=tmax, align=True)

ax2 = fig.add_subplot(122)

p = plotting(ax=ax2)
p.plot_record_section_SAC(st=st, phase=phase, tmin=tmin, tmax=tmax, align=True, type='baz')

plt.tight_layout()

plt.savefig("Record_Section.pdf")

plt.show()
