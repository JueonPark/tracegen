# input: ndpx_scheduling_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import os
import argparse
from xla_metadata_parser import parse_metadata

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--st', type=str, help="scheduling_table.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  original_results = open(args.st, 'r').read().split("\n")
  exp_path = os.getenv("EXP_PATH")
  output_name = "ndpx_scheduling_table_postprocessed.csv"
  output_path = os.path.join(exp_path, output_name)
  output = open(output_path, "w+")
  for original_row in original_results[:-1]:
    print(original_row)
    new_results = ""
    original_elements = original_row.split(",")

    new_results = original_elements[0] + "," + \
                  original_elements[1] + "," + \
                  original_elements[2] + "," + \
                  original_elements[3] + "," + \
                  original_elements[4] + "," + \
                  original_elements[5] + ","
    try:
      new_results += parse_metadata(original_elements[6])
    except:
      new_results += original_elements[6]
    new_results += "," + original_elements[7] + "," + \
                         original_elements[8] + ","
    try:
      new_results += parse_metadata(original_elements[9])
    except:
      new_results += original_elements[9]
    new_results += ",FALSE\n"
    output.write(new_results)