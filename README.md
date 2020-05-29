# Array_Seis_Circle
(work in progress) python package for array seismology methods correcting for a curved wavefront.

This is a work in progess and significant changes in the codes may occur, but these will be documented. I would suggest cloning this onto your machine and add the path to Circ_Array to
your $PYTHON_PATH for use. 

## Brief Descriptions of the directories and what is in them:

./Circ_Array/
  - Array_Plotting.py: class with plotting functions.
  - Circ_Array.py: class with utility array functions.
  - Circ_Beam.py: Numba compiled functions to do array processing
                  methods correcting for a curved wavefront.

./docs/
  - doc_Array_Circ.py: prints the doc string for the utilities module.
  - doc_Array_Plotting.py: prints the doc string for the plotting module.
  - doc_Circ_Beam.py: prints the doc string for the beamforming module.

./scripts/
  - Currently empty, but will include example scripts
    using the package for use in a work flow.

./tests/
  - Several testing scripts which also serve as example python codes using the package.
  - Example_uses.ipynb: Jupyter notebook showing uses of the package and functions.

./requirements.txt
  - text file of required python modules.

./setup.py
  - Currently only says to add the path to Circ_Array to your python path.
