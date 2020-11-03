#!/usr/bin/env python

from cluster_utilities import cluster_utilities

labels = [1]
points = [[0,0]]
c = cluster_utilities(labels=labels, points=points)

help(c)
