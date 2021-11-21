# Running trace
0. Get traces and hlo_graph
1. source setup_environment.sh
2. Change the address for the GPU kernels that **write** to NDPX kernel
    * change the address 0x7... to 0x1007 for those that have STG instruction.
    * use `replace_store_addr.py` for replacing address.
3. Put page table to the top of the GPU kernels that **read** from NDPX kernel.
4. Run