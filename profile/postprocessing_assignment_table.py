# input: ndpx_assignment_table.csv
# things to rewrite: NdpxOp, Layer
import argparse
from xla_metadata_parser import parse_metadata

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--at', type=str, help="assignment_table.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  original_results = open(args.at, 'r').read().split("\n")
  new_output_name = args.at.split(".")[0]
  new_output = open(new_output_name + "_postprocessed.csv", "w+")
  new_header = "NdpxOp,HopsFromRoot,NdpxKernelDimm,EstimatedNdpxCost,EstimatedGpuCost,Layer\n"
  new_output.write(new_header)
  for original_row in original_results:
    print(original_row)
    new_results = ""
    original_elements = original_row.split(",")
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
    new_output.write(new_results)