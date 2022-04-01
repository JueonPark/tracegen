# Scheduling traces
0. `source setup_environment.sh`
1. Use `copy_trace.py` to generate environment
2. Change the address for the GPU kernels that **write** to NDPX kernel
  * change the address `0x7...` to `0x1007` for those that have `STG` instruction.
  * use `replace_store_addr.py` for replacing address.
3. Put page table to the top of the GPU kernels that **read** from NDPX kernel.
  * This is needed for the case when the data is loaded from NDPX and stored in GPU on-the-fly.
  * use `attach_page_table.py` for replacing individual trace.
  * use `attach_page_table.sh` for replacing multiple traces.
4. You need to generate GPU_1, GPU_2, ... for cases that uses multi-gpu configuration. In that case, use `copy_gpu_folder.py` to copy scheduled `GPU_0` to `GPU_1`, `GPU_2`, ...
  * For specific trace, use `copy_specific_gpu_folder.py`.

# Running Trace
``` bash
vi sim_env_fw_jueon.sh
vi generate_simulation_env_fw_jueon.sh
vi sim_result_fw_jueon.sh
vi get_sim_result_fw_jueon.sh

# make environment and run script
sh sim_env_fw_jueon.sh
# run the jobs
sh sim_result_fw_jueon.sh
```
