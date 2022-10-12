#!/bin/bash
CONFIG=NDPX_baseline_64
TARGET_MODEL=$1
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
DEVICE_SETTING=GPU_${GPUS}_Buffer_${BUFFERS}
SIMULATOR_DIR=/home/jueonpark/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release

TRACE_DIR=`pwd`/traces/$1/exp_trace_dir
RESULT_DIR=`pwd`"/results/$DEVICE_SETTING/$TARGET_MODEL/"

CSV_FILES=`pwd`/csv_files
CSV_PATH=$CSV_FILES/$TARGET_MODEL-$CONFIG-$GPUS
if [ ${SYNC} -eq 1 ]; then
  CSV_PATH=$CSV_PATH-sync.csv
else
  CSV_PATH=$CSV_PATH-nosync.csv
fi
echo "GPUS,CONFIG,ID,NAME,CYCLE" > $CSV_PATH

# iterate and get each results
ls $RESULT_DIR | while read line 
do
  pushd $RESULT_DIR/$line/
  CYCLE_CNT=`cat GPU_0.out | grep -c "tot_sim_cycle"`
  CYCLE=`cat GPU_0.out | grep "sim_cycle" | head -n1 | awk '{print($3)}'` 
  CYCLE2=`cat GPU_0.out | grep "sim_cycle" | tail -n1 | awk '{print($3)}'` 
  NDP_CYCLE=`cat sim_result.out | grep "NDP kernel" | grep "launched" | awk '{print($11)}'`
  NDP_CYCLE2=`cat sim_result.out | grep "NDP kernel" | grep "finished" | awk '{print($11)}'`
  NAME=`cat GPU_0.out | grep "kernel_name" | head -n1 | awk '{print($3)}'` 
  JOB_NAME="${TARGET_MODEL}-GPU${3}-${line}-${CONFIG}"
		echo $JOB_NAME
  if [ ${SYNC} -eq 1 ]; then
    JOB_NAME="${JOB_NAME}-sync"
  else
    JOB_NAME="${JOB_NAME}-nosync"
  fi

  RUNNING=`squeue --format "%.200j %u %i" | grep -w $JOB_NAME`
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
    echo $DEVICE_SETTING,$CONFIG,$line,$NAME,$CYCLE " NOT FOUND" >> $CSV_PATH  ;
  else    
    if [ -n "$RUNNING" ]; then
        echo "POSSIBLE DEADLOCK $line"
    fi
    if [[ "$line" == *"_NDP_"* ]]; then
      echo $DEVICE_SETTING,$CONFIG,$line,$NAME,$CYCLE >> $CSV_PATH ;
      echo $DEVICE_SETTING,$CONFIG,$line,NDP_OP,$(( NDP_CYCLE2 - NDP_CYCLE )) >> $CSV_PATH ;
    else
      echo $DEVICE_SETTING,$CONFIG,$line,$NAME,$CYCLE >> $CSV_PATH ;
    fi
  fi
  popd
done