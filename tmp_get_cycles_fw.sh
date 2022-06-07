#!/bin/bash
Model=$1
KernelNum=$2

cat /home/jueonpark/tracegen/results/GPU_1_Buffer_1/$Model/NDPX_baseline_64/nosync/forward/$KernelNum/sim_result.out | grep "at cycle"
