# input: ndpx_scheduling_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import os
import pathlib
import argparse
from xla_metadata_parser import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

# The content consists of:
# NdpxKernel information
# - NdpxKernel
# - original instruction name
# - NdpxKernel's shape size
# - #input
# - #output
# - #op
# - estimated cost of NdpxKernel
# - layer(metadata) of NdpxKernel
# overlapping GPU kernel information
# - GpuKernel (basically GPU instruction's name)
# - pre-measured GpuKernel's cost
# - layer of GpuKernel
# execution option
# - on-the-fly or not (true if on-the-fly)
if __name__ == "__main__":
  args = parser.parse_args()
  exp_path = os.getenv("EXP_PATH")
  model = ""
  if (args.model).find("bert") != -1:
    model = "bert"
  elif (args.model).find("resnet") != -1:
    model = "resnet"
  elif (args.model).find("mobilenet") != -1:
    model = "mobilenet"
  elif (args.model).find("transformer") != -1:
    model = "transformer"
  elif (args.model).find("dlrm") != -1:
    model = "dlrm"
  elif (args.model).find("vit") != -1:
    model = "vit"
  else:
    exit(0)

  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  table_paths = list(xla_hlo_path.glob("*ndpx_scheduling_table*"))
  original_data = open(table_paths[0], "r").read().split("\n")[0] + "\n"
  for table_path in table_paths:
    table = open(table_path, "r").read()
    original_data += table.split("\n", 1)[1]
  original_results = original_data.split("\n")
  
  output_path = f'/home/jueonpark/tracegen/experiments_results/{args.model}/ndpx_scheduling_table_postprocessed.csv'
  output = open(output_path, "w+")


  for original_row in original_results[:-1]:
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
      if (model == "bert"):
        new_results += parse_bert_metadata(original_elements[7])
      elif (model == "dlrm"):
        new_results += parse_dlrm_metadata(original_elements[7])
      else:
        new_results += parse_mobilenet_metadata(original_elements[7])
    except:
      new_results += original_elements[7]
    new_results += "," + original_elements[8] + "," + \
                         original_elements[9] + ","
    try:
      if (model == "bert"):
        new_results += parse_bert_metadata(original_elements[10])
      elif (model == "dlrm"):
        new_results += parse_dlrm_metadata(original_elements[10])
      elif (model == "vit"):
        new_results += parse_dlrm_metadata(original_elements[10])
      else:
        new_results += parse_mobilenet_metadata(original_elements[10])
    except:
      new_results += original_elements[10]
    new_results += "," + original_elements[11] + "\n"
    output.write(new_results)