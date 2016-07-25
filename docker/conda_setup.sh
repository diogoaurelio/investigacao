#!/usr/bin/env bash

set -x -e

PYTHON_VERSION=3.4
SCRIPT_ENV="prod" # or stage

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
# make it available consistently for all hadoop sessions
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda config --set always_yes yes --set changeps1 no
conda update conda
conda create -n phd python=3.4
source activate phd
conda install jmespath botocore boto3 paramiko PyYAML requests numpy scipy scikit-learn pandas matplotlib psycopg2 seaborn dateutil botocore requests boto3
