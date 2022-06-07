#!/bin/bash
#PBS -P jm98
#PBS -N PROFILE_ALL
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=64GB
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/ka88
##qsub -I -qgpuvolta -lngpus=1 -lncpus=12 -lmem=64GB

module load tensorflow/2.3.0
export PYTHONPATH=$PYTHONPATH:/scratch/ka88/jg7534/tmp/lutAMDNN/

cd /scratch/ka88/jg7534/tmp/lutAMDNN/

./profile.sh
