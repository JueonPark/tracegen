# input: ndpx_candidate_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import os
import pathlib
import argparse
from xla_metadata_parser import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

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
  elif (args.model).find("transformer") != -1:
    model = "transformer"
  else:
    exit(0)

  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  table_paths = list(xla_hlo_path.glob("*ndpx_candidate_table*"))
  original_data = open(table_paths[0], "r").read().split("\n")[0] + "\n"
  for table_path in table_paths:
    table = open(table_path, "r").read()
    original_data += table.split("\n", 1)[1]
  original_results = original_data.split("\n")
  
  output_path = f'/home/jueonpark/tracegen/experiments_results/{args.model}/ndpx_candidate_table_postprocessed.csv'
  output = open(output_path, "w+")

  for original_row in original_results[:-1]:
    original_elements = original_row.split(",")
    layer_parsed = ""
    try:
      if model == "bert":
        layer_parsed = parse_bert_metadata(original_elements[1])
      elif model == "resnet":
        layer_parsed = parse_resnet_metadata(original_elements[1])
      elif model == "mobilenet":
        layer_parsed = parse_mobilenet_metadata(original_elements[1])
      elif model == "resnet":
        layer_parsed = parse_bert_metadata(original_elements[1])
      elif model == "dlrm":
        layer_parsed = parse_dlrm_metadata(original_elements[1])
      elif model == "vit":
        layer_parsed = parse_vit_metadata(original_elements[1])
      elif model == "transformer":
        layer_parsed = parse_transformer_metadata(original_elements[1])
    except:
      layer_parsed = original_elements[1]
    new_results = original_elements[0] + "," + \
                  layer_parsed + "," + \
                  original_elements[2] + "," + \
                  original_elements[3] + "," + \
                  original_elements[4] + "," + "\n"
    output.write(new_results)