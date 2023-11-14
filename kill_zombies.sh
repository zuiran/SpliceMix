#!/usr/bin/env bash

for gpu_id in $@; do

#  echo $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == 2 && $3 > 0 {print $3}')
  kill $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == '$gpu_id' && $3 > 0 {print $3}')
#  kill $(nvidia-smi -g '$gpu_id' | awk '$2=="Processes:" {p=1} p && $3 > 0 {print $3}')

done

# usage: ./kill_zombies.sh 0 1
# for killing processes of cuda devices 0, 1
