#!/bin/bash
KernelNum=$1
cat /home/jueonpark/tracegen/results/GPU_1_Buffer_1/220509_bert_large_cost_model_batch_2/NDPX_baseline_64/nosync/backward/$1/sim_result.out | grep "at cycle"