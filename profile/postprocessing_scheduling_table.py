# input: ndpx_scheduling_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import argparse
from xla_metadata_parser import parse_metadata

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--st', type=str, help="scheduling_table.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  original_results = open(args.st, 'r').read().split("\n")
  new_output_name = args.st.split(".")[0]
  new_output = open(new_output_name + "_postprocessed.csv", "w+")
  for original_row in original_results:
    print(original_row)
    new_results = ""
    original_elements = original_row.split(",")

    new_results = original_elements[0] + "," + \
                  original_elements[1] + "," + \
                  original_elements[2] + "," + \
                  original_elements[3] + "," + \
                  original_elements[4] + "," + \
                  original_elements[5] + "," + \
                  original_elements[6] + ","
    try:
      new_results += parse_metadata(original_elements[7])
    except:
      new_results += original_elements[7]
    new_results += "," + original_elements[8] + "," + \
                         original_elements[9] + "," + \
                         original_elements[10] + ","
    try:
      new_results += parse_metadata(original_elements[11])
    except:
      new_results += original_elements[11]
    new_results += ",FALSE\n"
    new_output.write(new_results)