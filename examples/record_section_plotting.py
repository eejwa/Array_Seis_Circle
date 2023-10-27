#!/usr/bin/env python

import obspy
import numpy as np
import matplotlib.pyplot as plt
from array_plotting import plotting

st = obspy.read('./data/19970525/*SAC')


fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(121)
p = plotting(ax)
p.plot_record_section_SAC(st=st, phase="SKS", tmin=-20, tmax=120, align=False, type='baz')
ax = fig.add_subplot(122)
p = plotting(ax)
p.plot_record_section_SAC(st=st, phase="SKS", tmin=-20, tmax=120, align=True, type='baz')
plt.show()


fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(121)
p = plotting(ax)
p.plot_record_section_SAC(st=st, phase="SKS", tmin=-20, tmax=120, align=False, type='distance')
ax = fig.add_subplot(122)
p = plotting(ax)
p.plot_record_section_SAC(st=st, phase="SKS", tmin=-20, tmax=120, align=True, type='distance')
plt.show()



