#!/bin/bash
Model=$1
KernelNum=$2

cat /home/jueonpark/tracegen/results/GPU_1_Buffer_1/$Model/$KernelNum/sim_result.out | grep "at cycle"
