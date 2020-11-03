# paramter file for the codes
# easier than having to repeatedly input flags for the codes

# path and wildcard for data
filepath='../tests/data/19970525/*SAC'

# Which 1D model to make predictions
model='prem'

# target phase
phase = 'SKS'

# phases of interest
phases = ['SKS','SKKS','sSKS','ScS','Sdiff','SS','pScS','sScS','pSKS', 'Pdiff']

# frequency band
fmin = 0.13
fmax = 0.52

# cut min/max for showing the record sections
cut_min = 150
cut_max = 150

# define slowness
slow_min = -2
slow_max = 2
s_space = 0.1

# define baz
baz_min = -30
baz_max = 30
b_space = 1

# PWS degree
degree=2

# do you want to manually pick the time window?
Man_Pick = True

# if not, what time window do you want to give?
t_min = 10
t_max = 20

# Do you want to align the traces?
Align = False

# what stacking do you want? linear (LIN), phase-weighted (PWS) or all (All)
Stack_type = 'Lin'

# Results_directory
Res_dir = "./Results/"
