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
  CSV_PATH=$CSV_PATH-sync-ndpx-cycle.csv
else
  CSV_PATH=$CSV_PATH-nosync-ndpx-cycle.csv
fi
echo "GPUS,CONFIG,SYNC,ID,NAME,CYCLE" > $CSV_PATH

# iterate and get each results
ls $RESULT_DIR | while read line 
do
  pushd $RESULT_DIR/$line/
  FINISHED=`cat sim_result.out | grep -c "Spent"`
  NDP_START_CYCLE=`cat sim_result.out | grep "launched NDP kernel" | awk '{print($11)}'`
  NDP_END_CYCLE=`cat sim_result.out | grep "finished NDP kernel" | awk '{print($11)}'`
  JOB_NAME="${TARGET_MODEL}-GPU${3}-${line}-${CONFIG}"
  if [ ${SYNC} -eq 1 ]; then
    JOB_NAME="${JOB_NAME}-sync"
  else
    JOB_NAME="${JOB_NAME}-nosync"
  fi

  RUNNING=`squeue --format "%.200j %u %i" | grep -w $JOB_NAME`
  if [ $FINISHED -ne 1 ]; then  
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
    echo $DEVICE_SETTING,$CONFIG,0,$line,NDP_OP,"NOT FOUND" >> $CSV_PATH  ;
  else
    if [ -n "$RUNNING" ]; then
        echo "POSSIBLE DEADLOCK $line"
    fi
    if [[ "$line" == *"_NDP_"* ]]; then
      echo $DEVICE_SETTING,$CONFIG,0,$line,NDP_OP,$(( NDP_END_CYCLE - NDP_START_CYCLE )) >> $CSV_PATH ;
    else
      continue
    fi
  fi
  popd
done