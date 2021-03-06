#!/bin/bash
packet_dir="xla_hlo/packet_32_"
for ((buffer=1; buffer<=4; buffer*=2))
do
  buf_dir="buffer_"$buffer"_"
  echo $buf_dir
  for ((gpu=1; gpu<=4; gpu*=2))
  do
    gpu_dir="gpu_"$gpu"_"
    echo $gpu_dir
    for ((sync=0; sync<=1; sync++))
    do
      sync_dir="sync_"$sync"_"
      echo $sync_dir
      mkdir -p $packet_dir$buf_dir$gpu_dir$sync_dir"simd_8"
    done
  done
done

mkdir -p output
