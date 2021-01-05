# paramter file for the codes
# easier than having to repeatedly input flags for the codes

# path and wildcard for data
filepath = "../tests/data/19970525/*SAC"

# Which 1D model to make predictions
pred_model = "prem"

# target phase
phase = "SKS"

# phases of interest
phases = ["SKS", "SKKS", "sSKS", "ScS", "Sdiff", "SS", "pScS", "sScS", "pSKS", "Pdiff"]

# frequency band
fmin = 0.13
fmax = 0.52

# cut min/max for showing the record sections
cut_min = 150
cut_max = 150

# define slowness box - relative
slow_min = -3
slow_max = 3
s_space = 0.05

# PWS degree
degree = 2

# do you want to manually pick the time window?
Man_Pick = False

# if not, what time window do you want to give?
t_min = 10
t_max = 30

# Do you want to align the traces?
Align = True

# what stacking do you want? linear (LIN), phase-weighted (PWS) or all (All)
Stack_type = "Lin"

# Results_dir
Res_dir = "./Results/"
