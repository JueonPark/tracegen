# Running NDPX simulations.
To cite NDPX, cite:

``` bibtex
@ARTICLE{9609620,  
  author={Ham, Hyungkyu and Cho, Hyunuk and Kim, Minjae and Park, Jueon and Hong, Jeongmin and Sung, Hyojin and Park, Eunhyeok and Lim, Euicheol and Kim, Gwangsun},  
  journal={IEEE Computer Architecture Letters},   
  title={Near-Data Processing in Memory Expander for DNN Acceleration on GPUs},   
  year={2021},  
  volume={20},  
  number={2},  
  pages={171-174},  
  doi={10.1109/LCA.2021.3126450}}
```

This repository provides a collection of scripts to run the NDPX experiments. To run NDPX experiment, you need the follwings:

 * `traces/` generated from NVBit,
 * `xla_hlo/` generated from NDPX compiler.

Put your model results on `./traces/{your-model}`. Remember that `{your-model}` must have both `traces/` and `xla_hlo/`.

# Experiments

## Running pattern-based scheduling for BERT
**WARNING - THIS IS ONLY AVAILABLE FOR BERT**

To run the pattern-matching-based experiment on BERT, do the following:

``` bash
./run_bert_experiment.sh {MODEL}
```

## Running one-by-one mapping experiment for ANY MODEL
From now, one-by-one mapping is the default experiment running script for NDPX experiment.

Just do the following:

``` bash
./run_overall_experiment.sh {MODEL}
```

## Running multiple overlapping experiment for ANY MODEL (EXPERIMENTAL)
Multiple overlapping is done by the scripts in `multiple_overlap_scheduler/`.

To run the multiple overlapping experiement, just do the following:

``` bash
./run_multiple_overlapping_experiment.sh {MODEL}
```

# Other Analysis Tools

## Postprocessing the results of the experiment
After finishing the experiment, we need to fetch the postprocessed data.

To get the data without NDPX cycle information, do:

`./run_after_experiment.sh {your-model}`

To get the data with NDPX cycle information, do:

`sim_result_jueon_full_ndp_cycle.sh {your-model}`

`./run_after_experiment.sh {your-model}`


## Calculating the overlapping cost
`ndpx_candidate_table.csv` contains the offloading decisions of NDPX-offloadable HLO instructions. It is inside `{your-model-path}/xla_hlo/`.

To analyze the cost of the overlapped NDPX kernels, do the following:

`python profile/calculate_candidate_table.py --model {your-model-path}`