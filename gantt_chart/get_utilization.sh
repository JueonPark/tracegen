#!/bin/bash
MODEL=$1
CONFIG=$2
GPUS=1
rm $MODEL-dir.out
SYNC=nosync
grep -l "kernel" ../results/GPU_${GPUS}_Buffer_${GPUS}/$MODEL/$CONFIG/*/*/*/sim_result.out | while read kernel;
do
#MAKE OFFCHIP LINK UTILIZATION
    DIR=`dirname $kernel`
    cat $kernel | grep "launched NDP kernel" | head -n1 | awk '{print $8}' > $DIR/NDP_name.out
    cat $DIR/sim_result.out | grep "Gantt" > $DIR/Gantt.out
    cat $DIR/sim_result.out | grep "CXL_0_ramulator_0 channel 4" > $DIR/NDPX_MEM_ch_3_UTIL.out
    cat $kernel | grep "UTIL GPU: 0Port: 0" > $DIR/GPU_0_Port_0_UTIL.out
#    cat $kernel | grep "CXL_0_ramulator_0 channel 0" > $DIR/NDPX_MEM_ch_3_UTIL.out 
    cat $kernel | grep "GPU_ramulator_6 channel "  > $DIR/GPU_MEM_ch_3_UTIL.out
    echo $DIR >> $MODEL-dir.out
done
