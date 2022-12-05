#!/bin/bash
# make sure that all the result files are properly made (no NOT FOUND!)
# - sh sim_result_jueon.sh
# - sh sim_result_dl_model_full_cycle.sh
# give model name as an argument
# - e.g. 220816_resnet18_fo_batch_32
if [ $# -eq 1 ]
then
  mkdir -p experiments_results/$1
  export EXP_PATH=/home/jueonpark/tracegen/experiments_results/$1

  # breakdown
  python HLO_breakdown/main.py --model $1
  python profile/offloadable_breakdown.py --model $1
  python profile/easy_op_breakdown.py --model $1

  # overlap scheduling
  # CAUTION: requires {args.model}-NDPX_baseline_64-1-nosync-full-ndp-cycle.csv to exist!
  # python profile/ndpx_result_merger.py --model $1

  python profile/postprocessing_candidate_table.py --model $1
  
  python profile/postprocessing_scheduling_table.py --model $1

  # after finishing all the jobs, compress to single zip file and 
  tar -cvf  $1-result.tar.gz experiments_results/$1
  mv $1-result.tar.gz ~/scp/
else
  echo "give experiment name(model name) as an argument: e.g. 220606_bert_large_overall_sim_exec_batch_2"
fi