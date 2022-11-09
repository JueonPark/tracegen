"""
Breakdowns the kernels into GEMM / cuDNN call / Fusion / Others.
"""
import csv
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  csv_path = f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'
  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  graph_paths = list(xla_hlo_path.glob("*after_optimizations.txt"))
  candidate_table_paths = list(xla_hlo_path.glob("ndpx_candidate*"))
  result_file = open(csv_path, "r")

  csv_result = csv.reader(result_file)
  header = next(csv_result)
  header.append("EASY_BREAKDOWN")
  output_path = f'/home/jueonpark/tracegen/experiments_results/{args.model}/{args.model}-NDPX_baseline_64-1-nosync_easy_breakdown.csv'
  csv_to_rewrite = csv.writer(open(output_path, "w+"))
  csv_to_rewrite.writerow(header)

  for row in csv_result:
    # 0               1                 2 3     4           5
    # GPU_1_Buffer_1	NDPX_baseline_64	0	8433	fusion_219	332801
    if row[4].find("fusion") != -1:
      row.append("XLA compution")
    elif row[4].find("broadcast") != -1:
      row.append("XLA compution")
    elif row[4].find("add") != -1:
      row.append("XLA compution")
    elif row[4].find("sub") != -1:
      row.append("XLA compution")
    elif row[4].find("mul") != -1:
      row.append("XLA compution")
    elif row[4].find("div") != -1:
      row.append("XLA compution")
    elif row[4].find("log") != -1:
      row.append("XLA compution")
    elif row[4].find("tanh") != -1:
      row.append("XLA compution")
    elif row[4].find("reduce") != -1:
      row.append("XLA compution")
    elif row[4].find("convert") != -1:
      row.append("XLA compution")
    elif row[4].find("gemm") != -1:
      row.append("cuBLAS GEMM")
    elif row[4].find("cu") != -1:
      row.append("cuDNN call")
    elif row[4].find("NDP_OP") != -1:
      row.append("NDPX")
    else:
      row.append("others")
    csv_to_rewrite.writerow(row)