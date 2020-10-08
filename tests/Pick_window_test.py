#!/usr/bin/env python

# code to test the manual pick time window function.

import obspy
import numpy as np
import time

from circ_array import circ_array
from circ_beam import BF_Spherical_XY_all, BF_Spherical_Pol_all
from array_plotting import plotting
c = Circ_Array()


st = obspy.read("./data/19990405/*SAC")
print(st[0].data)
phase = "SKS"
event_time = c.get_eventtime(st)
window = c.pick_tw(stream=st, phase=phase)

print(window)

rel_window = c.pick_tw(stream=st, phase=phase, align=True)
print(rel_window)
