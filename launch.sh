#!/usr/bin/env bash

flag=false
#count=0
for i in $@; do

  if [[ $i =~ "-"  ]]; then
    flag=false
  fi

  if $flag; then
    count=`expr $count + 1`
    cd+="$i,"
  fi

  if [ $i == -cd ]; then  # --cuda_devices is no longer available
    flag=true
    count=0
    cd=''
  fi

done

cd=${cd:0:`expr ${#cd} - 1`}
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=$cd
echo CUDA_VISIBLE_DEVICES="$cd"

python -m torch.distributed.launch --nproc_per_node $count main.py $@

# usage: ./lunch.sh -arg1 -arg2 ... -cd 0 1 -argn
# -cd denotes cuda devices, used here for counting nproc_per_node
# 'main_dist.py' in the last line should be replaced with your main file