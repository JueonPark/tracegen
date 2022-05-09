# Deep Learning Models (BERT)
Just do the followings:
1. `source setup_environment.sh`
2. `python BERT_NDPX/scripts/dependency_scheduler.py -m {model_name}_batch_{batch_size}`

    Remember to double-check the result so that both forward pass and backward pass are rightly configured.
    And you must copy `kernelslist.g.fw` and `kernelslist.g.bw` to `traces_fw` and `traces_bw` individually.
3. `python BERT_NDPX/scripts/post_process.py --model {model_name}_batch_{batch_size} --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all`
4. `python BERT_NDPX/scripts/make_expr_dir_hhk.py --model bert_base_cost_model_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all`
5. `sh sim_env_dl_model.sh`
6. `sh sim_result_dl_model.sh`

# Other Graphs
0. `source setup_environment.sh`
1. Use `copy_trace.py` to generate environment
2. Change the address for the GPU kernels that **write** to NDPX kernel
  * change the address `0x7...` to `0x1007` for those that have `STG` instruction.
  * use `replace_store_addr.py` for replacing address.
3. Put page table to the top of the GPU kernels that **read** from NDPX kernel.
  * This is needed for the case when the data is loaded from NDPX and stored in GPU on-the-fly.
  * use `attach_page_table.py` for replacing individual trace.
  * use `attach_page_table.sh` for replacing multiple traces.
<!-- 4. You need to generate GPU_1, GPU_2, ... for cases that uses multi-gpu configuration. In that case, use `copy_gpu_folder.py` to copy scheduled `GPU_0` to `GPU_1`, `GPU_2`, ...
  * For specific trace, use `copy_specific_gpu_folder.py`. -->
## Two directories
There are two directories for experiment
 * `traces/` preserves traces of various directories
 * `results/` stores run.sh files for each trace, and these are run by `sim_result_jueon.sh`

# Handling Baseline traces
TBD

# Running Trace
``` bash
# make environment and run script
sh sim_env_jueon.sh     # always modify the script first!
# run the jobs
sh sim_result_jueon.sh  # always modify the script first!
```
그냥 이걸 돌려라
``` bash
python BERT_NDPX/scripts/dependency_scheduler.py -m bert_base_cost_model_batch_16

python BERT_NDPX/scripts/post_process.py --model bert_base_cost_model_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

python BERT_NDPX/scripts/make_expr_dir_hhk.py --model bert_base_cost_model_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

sh sim_env_dl_model.sh
sh sim_result_dl_model.sh
```
