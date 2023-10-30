#!/usr/bin/env python

# code to test the manual pick time window function.

import obspy
from manual_pick import pick_tw

st = obspy.read("./data/19990405/*SAC")
phase = "SKS"

window = pick_tw(stream=st, phase=phase)

print(window)

rel_window = pick_tw(stream=st, phase=phase, align=True)
print(rel_window)
