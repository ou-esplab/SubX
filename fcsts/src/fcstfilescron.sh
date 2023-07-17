#!/bin/bash
set -xve

execDIR=/home/kpegion/projects/SubX/fcsts/src/
prog=mcssomfiletransfer.sh
fcstdate=$(date -dlast-Thursday +%Y%m%d)

cd $execDIR
./${prog} ${fcstdate} >& fcstfiles${fcstdate}.log
