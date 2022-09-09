#!/bin/sh
ml purge
ml Singularity
# singularity pull docker://location/omeslicc
singularity run -B "$(pwd)":/omeslicc omeslicc_latest.sif