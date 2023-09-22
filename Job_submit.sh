#!/bin/bash
FILENAME=$1
singularity exec --nv --bind /cfca-work /home/iwasakidk/docker/pytorch2201.sif env PYTHONUNBUFFERED=1 python /home/iwasakidk/VAE/"$FILENAME"