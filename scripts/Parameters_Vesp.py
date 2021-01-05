# paramter file for the codes
# easier than having to repeatedly input flags for the codes

# path and wildcard for data
filepath = "../tests/data/19970525/*SAC"

# Which 1D model to make predictions
model = "prem"

# target phase
phase = "SKS"

# phases of interest
phases = ["SKS", "SKKS", "sSKS", "ScS", "Sdiff", "SS", "pScS", "sScS", "pSKS", "Pdiff"]

# frequency band
fmin = 0.13
fmax = 0.52

# cut min/max for showing the record sections
cut_min = 50
cut_max = 200

# define slownesses to search over -
# will be shifted to be around predicted slowness
# target phase
smin = 2
smax = 10
s_step = 0.05

# define backazimuths to search over
bmin = 130
bmax = 180
b_step = 1

# PWS degree
degree = 2

# Slowness (slow) or Backazimuth (baz) vespagram?
Vesp_type = "slow"

# what stacking do you want? linear (LIN), phase-weighted (PWS) or all (All)
Stack_type = "Lin"

Res_dir = "./Results/"
