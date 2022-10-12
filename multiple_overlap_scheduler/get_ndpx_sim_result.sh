#!/bin/bash
CONFIG=NDPX_baseline_64
TARGET_MODEL=$1
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
SIMULATOR_DIR=/home/jueonpark/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release

TRACE_DIR=./traces/$1/exp_trace_dir
RESULT_DIR=./results/GPU_${GPUS}_Buffer_${BUFFERS}/$1
CSV_FILES=./csv_files
CSV_PATH=$CSV_FILES/$1-$CONFIG-$GPUS-nosync.csv

echo "GPUS,CONFIG,SYNC,ID,NAME,CYCLE" > $CSV_PATH
echo "NAME,LINES,SHAPE,CYCLE" > $CSV_PATH
ls $RESULT_DIR | while read line 
do
  pushd $RESULT_DIR/$line/
  CYCLE_CNT=`cat GPU_0.out | grep -c "tot_sim_cycle"`
  CYCLE=`cat sim_result.out | grep "NDP kernel" | grep "launched" | awk '{print($11)}'` 
  CYCLE2=`cat sim_result.out | grep "NDP kernel" | grep "finished" | awk '{print($11)}'` 
  JOB_NAME="${TARGET_MODEL}-GPU${3}-${line}-${CONFIG}"
		echo $JOB_NAME
  if [ ${SYNC} -eq 1 ]; then
    JOB_NAME="${JOB_NAME}-sync"
  else
    JOB_NAME="${JOB_NAME}-nosync"
  fi

  RUNNING=`squeue --format "%.200j %u %i" | grep -w $JOB_NAME`
  if [ $CYCLE_CNT -ne 1 ]; then  
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