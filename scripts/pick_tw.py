#!/usr/bin/env python

# code to test the manual pick time window function.

import obspy

import circ_array as c

st = obspy.read("./*sac")
phase = "SKS"

rel_window = c.pick_tw(stream=st, phase=phase, align=True)
print(rel_window)

with open('tw.txt', 'w') as tfile:
    tfile.write(f"{rel_window[0]} {rel_window[1]}")
