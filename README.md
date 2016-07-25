# Research

# Local Setup

## Python requirements

- Setup Anaconda. 
- Setup conda environment
- Setup local env

### Setup Anaconda
To download Anaconda package manager, go to: <i>https://www.continuum.io/downloads</i>.

After installing locally the conda environment, proceed to setup this project environment.


### Setup conda environment

For reliable updated requirements please check conda-requirements.txt Easiest is a conda environment:
```
conda create -n phd2 python=2.7
source activate phd2
conda install --file conda-requirements.txt
```

Note: At the moment of creation of this README, OpenCV is not available for python 3, thus the choice of python 2.7.

To deactivate this specific virtual environment:
```
source deactivate
```

Also another possibility is to use a docker container to forget package dependencies nightmare. <a href="https://hub.docker.com/r/flaviostutz/opencv-x86/">Here</a> is the one used in this project - many thanks to Fabio Stuzt.

