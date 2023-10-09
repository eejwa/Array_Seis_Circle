#import required functions
from setuptools import setup, find_packages

setup(
    author='James Ward',
    author_email='jamie.a.ward@hotmail.co.uk',
    description='A Python package for performing array seismology with corrections for curved wavefronts and automated beamforming grid searches.',
    name='circ_array',
    version='0.1.0',
    license='MIT',
    packages=find_packages(include=['circ_array.*']),
    scripts=['scripts/*'],
    install_requires =[
        'obspy==1.3.0',
        'numpy==1.23.3',
        'scipy==1.9.1',
        'matplotlib==3.5.3',
        'numba==0.55.2',
        'scikit-learn==1.1.2',
        'scikit-image==0.19.3'
        'pandas==1.4.4',
        'cartopy'
    ] ,
    python_requires='>=3.0.*'
)