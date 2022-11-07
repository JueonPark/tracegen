# input: ndpx_candidate_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import pathlib
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  
  xla_hlo_path_str = f'{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  table_paths = list(xla_hlo_path.glob("*ndpx_candidate_table*"))
  paralle_gpu_cost = 0
  paralle_ndpx_cost = 0
  sequential_gpu_cost = 0
  sequential_ndpx_cost = 0
  passed_gpu_cost = 0
  passed_ndpx_cost = 0
  for table_path in table_paths:
    table_df = pd.read_csv(table_path)
    table_df['GPUCost'] = pd.to_numeric(table_df['GPUCost'])
    table_df['NDPXCost'] = pd.to_numeric(table_df['NDPXCost'])
    try:
      pivot_table_df = pd.pivot_table(table_df, index="Decision",
                                      aggfunc=np.sum, fill_value=0)
    except:
      continue
    print(pivot_table_df)
    try:
      paralle_gpu_cost += pivot_table_df["GPUCost"]["PARALLEL"]
      paralle_ndpx_cost += pivot_table_df["NDPXCost"]["PARALLEL"]
    except:
      pass
    try:
      sequential_gpu_cost += pivot_table_df["GPUCost"]["ON_THE_FLY_READ"]
      sequential_ndpx_cost += pivot_table_df["NDPXCost"]["ON_THE_FLY_READ"]
    except:
      pass
    try:
      sequential_gpu_cost += pivot_table_df["GPUCost"]["ON_THE_FLY_WRITE"]
      sequential_ndpx_cost += pivot_table_df["NDPXCost"]["ON_THE_FLY_WRITE"]
    except:
      pass
    try:
      sequential_gpu_cost += pivot_table_df["GPUCost"]["SEQUENTIAL"]
      sequential_ndpx_cost += pivot_table_df["NDPXCost"]["SEQUENTIAL"]
    except:
      pass
    try:
      passed_gpu_cost += pivot_table_df["GPUCost"]["PASSED"]
      passed_ndpx_cost += pivot_table_df["NDPXCost"]["PASSED"]
    except:
      pass

  print(f"Total Parallel GPU Cost: {paralle_gpu_cost}")
  print(f"Total Parallel NDPX Cost: {paralle_ndpx_cost}")
  print(f"Total Sequential GPU Cost: {sequential_gpu_cost}")
  print(f"Total Sequential NDPX Cost: {sequential_ndpx_cost}")
  print(f"Total Passed GPU Cost: {passed_gpu_cost}")
  print(f"Total Passed NDPX Cost: {passed_ndpx_cost}")