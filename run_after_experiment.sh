#!/bin/bash
# make sure that all the result files are properly made (no NOT FOUND!)
# give model name as an argument
# - e.g. 220606_bert_large_overall_sim_exec_batch_2
if [ $# -eq 1 ]
then
  mkdir experiments_results/$1
  export EXP_PATH=/home/jueonpark/tracegen/experiments_results/$1
  python HLO_breakdown/main.py --csv csv_files/$1-NDPX_baseline_64-1-nosync-fw.csv --hlo traces/$1/xla_hlo/$1.txt
  python HLO_breakdown/main.py --csv csv_files/$1-NDPX_baseline_64-1-nosync-bw.csv --hlo traces/$1/xla_hlo/$1.txt

  # device assignment
  python profile/postprocessing_assignment_table.py --at traces/$1/xla_hlo/ndpx_device_assigner_assignment_table_cluster_0.csv
  python profile/postprocessing_subgraph_table.py  --st traces/$1/xla_hlo/ndpx_device_assigner_subgraph_table_cluster_0.csv

  # overlap scheduling
  python profile/ndpx_result_merger.py --fw csv_files/$1-NDPX_baseline_64-1-nosync-fw.csv --bw csv_files$1-NDPX_baseline_64-1-nosync-bw.csv --kfw traces/$1/traces_fw/kernelslist.g.fw --kbw traces/$1/traces_bw/kernelslist.g.bw --ne traces/$1/xla_hlo/ndpx_scheduling_table_cluster_0.csv
  python profile/postprocessing_scheduling_table.py --st traces/$1/xla_hlo/ndpx_scheduling_table_cluster_0.csv
else
  echo "give experiment name(model name) as an argument"
fi