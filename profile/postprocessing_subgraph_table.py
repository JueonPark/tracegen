# input: ndpx_assignment_table.csv
# things to rewrite: NdpxOp, Layer
import os
import argparse
from xla_metadata_parser import parse_metadata

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--st', type=str, help="subgraph_table.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  original_results = open(args.st, 'r').read().split("\n")
  exp_path = os.getenv("EXP_PATH")
  output_name = "ndpx_subgraph_table_postprocessed.csv"
  output_path = os.path.join(exp_path, output_name)
  output = open(output_path, "w+")
  for original_row in original_results[:-1]:
    print(original_row)
    new_results = ""
    original_elements = original_row.split(",")
    if original_elements[0].find("$") != -1:
      new_results += original_elements[0].split("$")[1]
    else:
      try:
        new_results += original_elements[0].split(".")[0]
      except:
        new_results += original_elements[0]
    new_results += "," + original_elements[1] + "," \
                       + original_elements[2] + "," \
                       + original_elements[3] + "," \
                       + original_elements[4] + ","
    try:
      new_results += parse_metadata(original_elements[5])
    except:
      new_results += original_elements[5]
    new_results += "\n"
    output.write(new_results)