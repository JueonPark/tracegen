# calculate the total offline execution result
import pathlib
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--issim", type=bool, default=False)

if __name__ == "__main__":
  args = parser.parse_args()
  xla_hlo_path_str = f'{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  table_paths = list(xla_hlo_path.glob("*offline_execution_result*"))

  total_gpu_cost = 0
  for table_path in table_paths:
    table_df = pd.read_csv(table_path)
    gpu_cost = table_df.sum()["runtime"]
    if not args.issim:
      gpu_cost *= 0.002274991
    total_gpu_cost += (int)(gpu_cost)

  print(f"Total GPU Cost: {total_gpu_cost}")