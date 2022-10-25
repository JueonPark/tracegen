#!/bin/bash
CONFIG=$1
TARGET_MODEL=$2
GPUS=$3
BUFFERS=$4
SYNC=$5
RESUBMIT=$6
echo $SYNC
DEVICE_SETTING=GPU_${GPUS}_Buffer_${BUFFERS}
SIMULATOR_DIR=/home/jueonpark/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release
TARGET_TRACE_DIR=`pwd`/traces/$TARGET_MODEL
TARGET_RESULT_DIR=`pwd`"/results/"$DEVICE_SETTING"/"$TARGET_MODEL"/"$CONFIG
if [ ${SYNC} -eq 1 ]; then
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"sync
else
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"nosync
fi
CSV_FILES=`pwd`/csv_files
CSV_PATH=$CSV_FILES/$TARGET_MODEL-$CONFIG-$3
if [ ${SYNC} -eq 1 ]; then
    CSV_PATH=$CSV_PATH-sync.csv
else
    CSV_PATH=$CSV_PATH-nosync.csv
fi
echo "GPUS,CONFIG,SYNC,ID,NAME,CYCLE" > $CSV_PATH
ls $TARGET_RESULT_DIR | while read line 
do
    pushd $TARGET_RESULT_DIR/$line/
    CYCLE_CNT=`cat GPU_0.out | grep -c "tot_sim_cycle"`
    CYCLE=`cat GPU_0.out | grep "sim_cycle" | head -n1 | awk '{print($3)}'` 
    CYCLE2=`cat GPU_0.out | grep "sim_cycle" | tail -n1 | awk '{print($3)}'` 
    NAME=`cat GPU_0.out | grep "kernel_name" | head -n1 | awk '{print($3)}'` 
    JOB_NAME="${TARGET_MODEL}-GPU${3}-${line}-${CONFIG}"
		echo $JOB_NAME

    if [ ${SYNC} -eq 1 ]; then
        JOB_NAME="${JOB_NAME}-sync"
    else
        JOB_NAME="${JOB_NAME}-nosync"
    fi
    RUNNING=`squeue --format "%.200j %u %i" | grep -w $JOB_NAME`
    echo $RUNNING
    if [ $CYCLE_CNT -ne 2 ]; then  
        if [ -z "$RUNNING" ]; then
            echo "RUNNING NOT FOUND ${line}" 
            if [ "$RESUBMIT" = "1" ]; then
                rm GPU*
                # run job
                sbatch -J $JOB_NAME -n 1 --partition=allcpu -o sim_result.out -e sim_result.err run.sh;
            fi
        fi
        echo "NOT FOUND $line"
        echo `cat sim_result.out | tail -n1` 
        echo $DEVICE_SETTING,$CONFIG,$5,$line,$NAME,$CYCLE " NOT FOUND" >> $CSV_PATH  ;
    else    
        if [ -n "$RUNNING" ]; then
            echo "POSSIBLE DEADLOCK $line"
        fi
        echo $DEVICE_SETTING,$CONFIG,$5,$line,$NAME,$CYCLE >> $CSV_PATH ;
        echo $DEVICE_SETTING,$CONFIG,$5,$line,NDP_OP,$(( CYCLE2 - CYCLE )) >> $CSV_PATH ;
    fi
    popd
done