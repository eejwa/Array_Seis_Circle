# paramter file for the codes
# easier than having to repeatedly input flags for the codes

# path and wildcard for data
# filepath='../tests/data/19970525/*SAC'
filepath='../tests/data/19970525/*SAC'

# Which 1D model to make predictions
pred_model='prem'

# target phase
phase = 'SKS'

# phases of interest
phases = ['SKS','SKKS','sSKS','ScS','Sdiff','SS','pScS','sScS','pSKS', 'Pdiff']
# phases = ['SKS','SKKS']

# Do you want to filter?
Filt = True

# Results directory path
Res_dir = "./Results/numpy_arrays/"

# frequency band
fmin = 0.10
fmax = 0.40

# cut min/max - for some reason the aligning function does not
# work if the traces are too long
cut_min = 150
cut_max = 150

# define slowness box - relative
slow_min = -3
slow_max = 3
s_space = 0.05

# time window to define around the traces
# aligned on the predicted slowness of the target phase
t_min = 0
t_max = 30

# Do you want to align the traces?
Align = True

# Number Bootstrap samples
Boots = 250

# what to multiply noise estimate by
threshold_multiplier = 3

# number of peaks above threshold to take
peak_number = 3


# DBSCAN parameters
epsilon = 0.2
MinPts = 0.15
