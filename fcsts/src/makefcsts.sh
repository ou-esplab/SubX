#!/bin/bash
set -xve

# Load python module to get access to conda
. /usr/share/Modules/init/bash
#module load anaconda/3

# Activate conda environment
. /home/mlavoie/miniconda3/etc/profile.d/conda.sh
conda activate SubX

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

# Get the fcstdate provided as command line argument
fcstdate=$1
echo ${fcstdate}

# Set Output Path
#outPath=/shared/subx/forecast/weekly/
outPath=/share/scratch/kpegion/subx/forecast/weekly/

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

# Compress data 
#for var in pr_sfc rlut_toa tas_2m ts_sfc ua_200 ua_850 va_200 va_850 zg_200 zg_500 uas_10m vas_10m psl_
#for var in pr_sfc tas_2m zg_500
#do

#/usr/bin/ncks -h -O -4 --dfl_lvl=1 --cnk_plc=g2d --cnk_dmn lat,181 --cnk_dmn lon,360 --cnk_dmn time,1 ${outPath}${fcstdate}/data/fcst_${fcstdate}.anom.${var}.nc ${outPath}${fcstdate}/data/fcst_${fcstdate}.anom.${var}.nc4
# cp ${outPath}/${fcstdate}/data/fcst_${fcstdate}.anom.${var}.nc4 ${scratchDir}/fcst_${fcstdate}.anom.${var}.nc

# case $var in
#    pr_sfc ) varjma=pr ;;
#   rlut_toa ) varjma=olr ;;
#    tas_2m ) varjma=tas ;;
#    ts_sfc ) varjma=sst ;;
#    ua_200 ) varjma=u200 ;;
#    ua_850 ) varjma=u850 ;;
#    va_200 ) varjma=v200 ;;
#    va_850 ) varjma=v850 ;;
#    zg_200 ) varjma=z200 ;;
#    zg_500 ) varjma=z500 ;;
#    uas_10m ) varjma=u10m ;;
#    vas_10m ) varjma=v10m ;;
#    psl ) varjma=mslp ;;
#  esac
#
#done

# Remove lockfile if this program runs to completion
rm ${outPath}/$fcstdate/subxfcst.lock

