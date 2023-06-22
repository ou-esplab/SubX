#!/bin/bash
set -xve
fcstdate=$1
sourceDir=/share/scratch/kpegion/subx/forecast/weekly/${fcstdate}/
destDir=/home/kpegion/http/subx/forecasts/
destHost=somclass23.som.nor.ou.edu
execDir=/home/kpegion/projects/SubX_Fcsts/src

# Check if lock file exists if so, wait 1 min and check again
# Timeout after 1 hour

timeout=60
while [ $timeout > 0 ] && [ -f ${sourceDir}/subxfcst.lock ]
do
  sleep 60
  ((timeout -= 1))
done

# If we didnt timeout then continue, otherwise exit with error
if [ $timeout >  0 ]
then

    # Create Forecast Date Directories on somclass

    ssh ${destHost} "mkdir -p ${destDir}/images/${fcstdate}"
    ssh ${destHost} "mkdir -p ${destDir}/data/${fcstdate}"

    # Copy images and data to appropriate directories on somclass

    scp ${sourceDir}/images/${fname}/* ${destHost}:${destDir}/images/${fcstdate}/
    scp ${sourceDir}/data/${fname}/* ${destHost}:${destDir}/data/${fcstdate}/
    scp ${sourceDir}/images/${fname}/* ${destHost}:${destDir}/images/Latest/

    # Run Python Program to update html on somclass
#    scp ${destHost}:${destDir}/forecasts.html ./forecasts.${fcstdate}.html
#    ./updatehtmldates.py --date ${fcstdate}
#    scp output.${fcstdate}.html ${destHost}:${destDir}/forecasts.html
else

    echo "ERROR: makesubsfcsts.sh did not complete successfully. Lock file still present for $fcstdate"
    exit 1 # terminate and indicate error

fi
