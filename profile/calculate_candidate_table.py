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
  gpu_cost = 0
  ndpx_cost = 0
  for table_path in table_paths:
    table_df = pd.read_csv(table_path)
    table_df['GPUCost'] = pd.to_numeric(table_df['GPUCost'])
    table_df['NDPXCost'] = pd.to_numeric(table_df['NDPXCost'])
    pivot_table_df = pd.pivot_table(table_df, index="Decision",
                                    aggfunc=np.sum, fill_value=0)
    print(pivot_table_df)
    gpu_cost += pivot_table_df["GPUCost"]["NOT_ON_THE_FLY"]
    ndpx_cost += pivot_table_df["NDPXCost"]["NOT_ON_THE_FLY"]

  print(f"GPU Cost: {gpu_cost}")
  print(f"NDPX Cost: {ndpx_cost}")