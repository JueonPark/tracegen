# Running trace
0. Get traces and hlo_graph
1. source setup_environment.sh
2. Change the address for the GPU kernels that **write** to NDPX kernel
    * change the address 0x7... to 0x1007 for those that have STG instruction.
    * use `replace_store_addr.py` for replacing address.
3. Put page table to the top of the GPU kernels that **read** from NDPX kernel.
4. Run

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