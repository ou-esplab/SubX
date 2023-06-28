#!/bin/bash
set -xve

hostsub=$(echo $HOSTNAME | cut -c 1-5)
echo $hostsub

#execDIR=/home/mlavoie/SubX_Fcsts/src/
execDIR=/home/kpegion/projects/SubX/fcsts/src/
prog=makefcsts.sh
fcstdate=$(date -dlast-Thursday +%Y%m%d)
cd $execDIR
./${prog} ${fcstdate} &> fcst.log

