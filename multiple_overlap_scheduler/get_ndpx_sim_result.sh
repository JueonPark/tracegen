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

if [ -z $TARGET_MODEL ]
then
  exit 0
fi

TRACE_DIR=`pwd`/traces/$1/exp_trace_dir
RESULT_DIR=`pwd`"/results/$DEVICE_SETTING/$TARGET_MODEL/"

CSV_FILES=`pwd`/csv_files
CSV_PATH=$CSV_FILES/$TARGET_MODEL-$CONFIG-$GPUS
if [ ${SYNC} -eq 1 ]; then
  CSV_PATH=$CSV_PATH-sync.csv
else
  CSV_PATH=$CSV_PATH-nosync.csv
fi
echo "GPUS,CONFIG,SYNC,ID,NAME,CYCLE" > $CSV_PATH

# iterate and get each results
ls $RESULT_DIR | while read line 
do
  pushd $RESULT_DIR/$line/
  FINISHED=`cat sim_result.out | grep -c "Spent"`
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
    echo $DEVICE_SETTING,$CONFIG,0,$line,$NAME,$CYCLE " NOT FOUND" >> $CSV_PATH  ;
  else
    if [ -n "$RUNNING" ]; then
        echo "POSSIBLE DEADLOCK $line"
    fi
    if [[ "$line" == *"custom-call"* ]]; then
      TOTAL_NAMES=`cat GPU_0.out | grep "kernel_name"`
      echo $TOTAL_NAMES
      delimiter="kernel_name = "
      s=$TOTAL_NAMES$delimiter  
      NAMES=();  
      while [[ $s ]];  
      do  
      NAMES+=( "${s%%"$delimiter"*}" );  
      s=${s#*"$delimiter"};  
      done;  
      declare -p NAMES  
      echo "names: $NAMES"
      for NAME in ${NAMES[@]}
      do
        if [ -z $NAME ]
        then
          continue
        fi
        echo "name: $NAME"
        START_CYCLE=`cat sim_result.out | grep "$NAME at cycle" | head -n1 | awk '{print($11)}'`
        END_CYCLE=`cat sim_result.out | grep "$NAME at cycle" | tail -n1 | awk '{print($11)}'`
        echo $DEVICE_SETTING,$CONFIG,0,$line,$NAME,$(( END_CYCLE - START_CYCLE )) >> $CSV_PATH ;
      done
      NDP_START_CYCLE=`cat sim_result.out | grep "launched NDP kernel" | awk '{print($11)}'`
      NDP_END_CYCLE=`cat sim_result.out | grep "finished NDP kernel" | awk '{print($11)}'`
      NDP_CYCLE=$(( CYCLE2 > NDP_END_CYCLE ? 0 : NDP_END_CYCLE - CYCLE2 ))
      echo $DEVICE_SETTING,$CONFIG,0,$line,NDP_OP,$NDP_CYCLE >> $CSV_PATH ;
    else
      echo $DEVICE_SETTING,$CONFIG,0,$line,$NAME,$CYCLE >> $CSV_PATH ;
    fi
  fi
  popd
done
