#!/usr/bin/env python

import obspy
import numpy as np

from array_plotting import plotting
p = plotting()

st = obspy.read('./data/19990405/*SAC')

p.plot_record_section_SAC(st=st, phase="SKS", tmin=20, tmax=120, align=False)

p.plot_record_section_SAC(st=st, phase="SKS", tmin=20, tmax=120, align=True)
