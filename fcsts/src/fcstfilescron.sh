#!/bin/bash
set -xve

execDIR=/home/kpegion/projects/SubX_Fcsts/src/
prog=mcssomfiletransfer.sh
fcstdate=$(date -dlast-Thursday +%Y%m%d)

cd $execDIR
./${prog} ${fcstdate} >& fcstfiles${fcstdate}.log
