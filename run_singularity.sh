#!/bin/sh
ml purge
ml Singularity
singularity run -B "$(pwd)":/omeslicc omeslicc_latest.sif