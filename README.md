# Array_Seis_Circle
Python package for array seismology methods correcting for a curved wavefront.

The code is currently being improved and functions added regularly, therefore there may still be some bugs in the code. This is mainly a result of me using this for specific purposes. There will be additional utilities added to this package as time goes on.

I would suggest cloning this onto your machine and add the path to Circ_Array to your $PYTHON_PATH for use.

## Brief Descriptions of the directories and what is in them:

### ./circ_array/
  - array_plotting.py: class with plotting functions.
  - circ_array.py: class with utility array functions.
  - circ_beam.py: Numba compiled functions to do array processing
                  methods correcting for a curved wavefront.
  - cluster_utilities.py: class with functions to calculate the
                          properties of clusters found from
                          DBSCAN.

### ./docs/
  - doc_Array_Circ.py: prints the doc string for the utilities module.
  - doc_Array_Plotting.py: prints the doc string for the plotting module.
  - doc_Circ_Beam.py: prints the doc string for the beamforming module.
  - doc_Cluster_Utilities.py: prints the doc string for the cluster utilities class.


### ./scripts/
  Several python codes which utilise functions in the module to perform analysis.
  These act as out of the box tools for users to use and edit to their own purposes.
  The current tools include:

  - Bootstrap_Peak_Recover_XY.py
    - Bootstrap samples the waveforms and recovers potential slowness vectors via beamforming
      and estimating a noise value.
      - Takes parameters from Parameters_Bootstrap.py.
      - Can be run over multiple cores with `$ mpirun -np N Bootstrap_Peak_Recover_XY.py`,
        where N is the number of cores.
      - These are stored in a numpy array in a results directory defined by the user.

  - Clustering.py
    - Using the output of Bootstrap_Peak_Recover_XY.py, perform DBSCAN with MinPts
      and $\epsilon$ defined in Parameters_Bootstrap.py.
      - Will plot the resulting clusters and noise with the mean of all the $\theta-p$
        plots found from each bootstrap sample.

  - TP_XY.py
    - Performs a beamforming grid search correcting for a curved wavefront describing slowness vectors in cartesian
      coordinates.
      - Input parameters are taken from "Parameters_TP_XY.py".
      - Can stack either linearly or phase weighted stacking.
      - Currently, the code targets a particular phase such as SKS. The predicted
        arrival time needs to be in one of the sac headers tn and the phase name
        in the associated ktn.
      - Results of backazimuth and horizontal slowness deviations are stored in the
        results directory defined in the parameter file.
      - Can manually pick the time window or give times relative to predicted arrival
        time of target phase.
      - Can perform a relative beamforming process by aligning on the horizontal slowness
        of the target phase.

  - TP_Pol.py
    - Performs a beamforming grid search correcting for a curved wavefront describing slowness vectors in polar coordinates.
      - Input parameters are taken from "Parameters_TP_Pol.py".
      - Can stack either linearly or phase weighted stacking.
      - Currently, the code targets a particular phase such as SKS. The predicted
        arrival time needs to be in one of the sac headers tn and the phase name
        in the associated ktn.
      - Results of backazimuth and horizontal slowness deviations are stored in the
        results directory defined in the parameter file.
      - Can manually pick the time window or give times relative to predicted arrival
        time of target phase.
      - Can perform a relative beamforming process by aligning on the horizontal slowness
        of the target phase.
  - Vesp.py
    - Creates a slowness or backazimuth vespagram with either linear or phase weighted stacking.
      - The arrival time for the target phase needs to be stored in the SAC headers tn and the phase name in ktn.  

### ./tests/
  - Several testing scripts which also serve as example python codes using the package.
  - Example_uses.ipynb: Jupyter notebook showing uses of the package and functions.

### Installation
  - I recommend using [anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the python packages.
  - The conda environment can be created using the conda environment yml file by:
    - ```$ conda env create -f Bootstrap_Cluster.yml ```
  - Then this environment can be activated by:
    - ```conda activate Boots_Cluster ```

  - Because this package is likely to change quickly in the near future, I recommend adding
    the path to the circ_array directory to your python path. This will mean the modules can be imported anywhere via python.

### ./setup.py
  - Currently only says to add the path to Circ_Array to your python path.
