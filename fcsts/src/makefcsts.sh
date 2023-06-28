#!/bin/bash
set -xve

# Load python module to get access to conda
#. /usr/share/Modules/init/bash
#module load anaconda/3

# Activate conda environment
. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate subx

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

# Get the fcstdate provided as command line argument
fcstdate=$1
echo ${fcstdate}

# Set Output Path
#outPath=/shared/subx/forecast/weekly/
#outPath=/share/scratch/kpegion/subx/forecast/weekly/
outPath=/data/esplab/shared/subx/forecast/weekly/

# Make directories for this forecast if they don't exist
if [ ! -d "${outPath}/$fcstdate/images/" ]
then
   echo "Making Directory ${outPath}/$fcstdate/images/"
   mkdir -p ${outPath}/$fcstdate/images
fi
if [ ! -d "${outPath}/$fcstdate/data/" ]
then
   echo "Making Directory ${outPath}/$fcstdate/data/"
   mkdir -p ${outPath}/$fcstdate/data
fi

# Create lock file in this fcsts directory
touch ${outPath}/$fcstdate/subxfcst.lock

# Run Program to Make Forecast plot and data files
./MakeSubXFcst.py --date ${fcstdate} 

# Remove lockfile if this program runs to completion
rm ${outPath}/$fcstdate/subxfcst.lock

