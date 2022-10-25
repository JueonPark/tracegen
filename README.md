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

## Quick Start
Just do the following:

``` bash
./run_overall_experiment.sh {MODEL}
```

## Running pattern-based scheduling for BERT
**WARNING - THIS IS ONLY AVAILABLE FOR BERT**

To run the pattern-matching-based experiment on BERT, do the following:

`./run_bert_experiment.sh {MODEL}`.

## Running one-by-one mapping experiment for ANY MODEL
From now, one-by-one mapping is the default experiment running script for NDPX experiment.

Just do `./run_overall_experiment.sh {MODEL}`.

## Running multiple overlapping experiment for ANY MODEL (EXPERIMENTAL)
Multiple overlapping is done by the scripts in `multiple_overlap_scheduler/`.

To run the multiple overlapping experiement, do the followings:

1. `python multiple_overlap_scheduler/make_ndpx_sim_dir.py --model {your-model}`

 * This would schedule the traces and generate the trace directories.

2. `sh multiple_overlap_scheduler/make_ndpx_sim_env.sh {MODEL}`

 * This would generate the simulation environment.

3. `sh multiple_overlap_scheduler/get_ndpx_sim_result.sh {MODEL}`

 * This would run the simulation.