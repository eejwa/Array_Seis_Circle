#!/usr/bin/env python


Description = """
This python script will test:
    - Estimating the arrival times of a curved and plan wavefront to a seismic array.
    - Compare these to the actual arrival times (from PREM).
"""

# imports
import obspy
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from obspy.taup import TauPyModel
model = TauPyModel(model="prem")

from circ_array import circ_array
from circ_beam import calculate_time_shifts
c = circ_array()

# parameters
# phase of interest
target_phase = 'SKS'
other_phases = ['SKS','SKKS','ScS','Sdiff','sSKS','sSKKS','PS']


# read in data
st = obspy.read('./data/19990405/*SAC')
evdp = st[0].stats.sac.evdp
evla = st[0].stats.sac.evla
evlo = st[0].stats.sac.evlo


# get array metadata
event_time = c.get_eventtime(st)
geometry = c.get_geometry(st)
distances = c.get_distances(st,type='deg')
mean_dist = np.mean(distances)
stations = c.get_stations(st)

centre_x = np.mean(geometry[:,0])
centre_y = np.mean(geometry[:,1])

# get data
Traces = c.get_traces(st)

# get predicted slowness and backazimuth
predictions = c.pred_baz_slow(
    stream=st, phases=[target_phase], one_eighty=True)

# find the line with the predictions for the phase of interest
row = np.where((predictions == target_phase))[0]
P, S, BAZ, PRED_BAZ_X, PRED_BAZ_Y, PRED_AZ_X, PRED_AZ_Y, DIST, TIME = predictions[row, :][0]

S = float(S)
BAZ = float(BAZ)

shifts_circ, times_circ = calculate_time_shifts(traces=Traces, geometry=geometry,
                                      abs_slow=S, baz=BAZ, distance=mean_dist,
                                      centre_x=centre_x, centre_y=centre_y,
                                      type='circ')

shifts_plane, times_plane = calculate_time_shifts(traces=Traces, geometry=geometry,
                                      abs_slow=S, baz=BAZ, distance=mean_dist,
                                      centre_x=centre_x, centre_y=centre_y,
                                      type='plane')

taup_times = []
for d in distances:
    arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=d,
                                  phase_list=[target_phase])

    taup_times.append(arrivals[0].time)

true_times = np.array(taup_times) - float(TIME)

print(mse(true_times, times_circ))
print(mse(true_times, times_plane))
