#!/bin/bash
# make sure that all the result files are properly made (no NOT FOUND!)
# - sh sim_result_dl_model.sh
# - sh sim_result_dl_model_full_cycle.sh
# give model name as an argument
# - e.g. 220606_bert_large_overall_sim_exec_batch_2
if [ $# -eq 1 ]
then
  mkdir -p experiments_results/$1
  export EXP_PATH=/home/jueonpark/tracegen/experiments_results/$1

  # merge fw and bw
  python profile/fw_bw_merger.py --fw csv_files/$1-NDPX_baseline_64-1-nosync-fw.csv \
                                 --bw csv_files/$1-NDPX_baseline_64-1-nosync-bw.csv \
                                 --ffc csv_files/$1-NDPX_baseline_64-1-nosync-fw-ndp-full-cycle.csv \
                                 --bfc csv_files/$1-NDPX_baseline_64-1-nosync-bw-ndp-full-cycle.csv \
                                 --on $1

  # breakdown
  python HLO_breakdown/main.py \
      --csv $EXP_PATH/$1.csv \
      --hlo traces/$1/xla_hlo/after_optimizations.mlir
  python HLO_breakdown/main.py \
      --csv $EXP_PATH/$1-ndp-full-cycle.csv \
      --hlo traces/$1/xla_hlo/after_optimizations.mlir

  # # device assignment
  # # python profile/postprocessing_assignment_table.py \
  # #         --at traces/$1/xla_hlo/ndpx_device_assigner_assignment_table_cluster_0.csv
  # # python profile/postprocessing_subgraph_table.py \
  # #         --st traces/$1/xla_hlo/ndpx_device_assigner_subgraph_table_cluster_0.csv

  # overlap scheduling
  python profile/ndpx_result_merger.py --csv $EXP_PATH/$1-ndp-full-cycle.csv \
                                       --kfw traces/$1/traces_fw/kernelslist.g.fw \
                                       --kbw traces/$1/traces_bw/kernelslist.g.bw \
                                       --ne traces/$1/xla_hlo/ndpx_scheduling_table_cluster_0.csv

  python profile/postprocessing_candidate_table.py --ct traces/$1/xla_hlo/ndpx_candidate_table_cluster_0.csv
  python profile/postprocessing_scheduling_table.py --st traces/$1/xla_hlo/ndpx_scheduling_table_cluster_0.csv

	# for three encoder bert large, make it to full 24 layer BERT Large result csv file.
	python profile/construct_24_encoder_results.py \
			--csv $EXP_PATH/"$1"_breakdown.csv \
			--hlo traces/$1/xla_hlo/after_optimizations.mlir \
      --kfw traces/$1/traces_fw/kernelslist.g.fw \
      --kbw traces/$1/traces_bw/kernelslist.g.bw \

  python profile/refine_result.py --csv $EXP_PATH/"$1"_24layer_breakdown.csv

  # after finishing all the jobs, compress to single zip file and 
  tar -cvf  $1-result.tar.gz experiments_results/$1
  mv $1-result.tar.gz ~/scp/
else
  echo "give experiment name(model name) as an argument: e.g. 220606_bert_large_overall_sim_exec_batch_2"
fi