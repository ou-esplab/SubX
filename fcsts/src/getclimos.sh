#!/bin/bash
set -xve

# Load python module to get access to conda
. /usr/share/Modules/init/bash
#module load anaconda/3

# Activate conda environment
. /home/kpegion/miniconda3/etc/profile.d/conda.sh
conda activate subx

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

# Run Program to Make Forecast plot and data files
./getClimos.py

