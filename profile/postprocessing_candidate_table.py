# input: ndpx_candidate_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import os
import argparse
from xla_metadata_parser import *

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--ct', type=str, help="candidate_table.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  original_results = open(args.ct, 'r').read().split("\n")
  exp_path = os.getenv("EXP_PATH")
  model = ""
  if exp_path.find("bert") != -1:
    model = "bert"
  elif exp_path.find("resnet") != -1:
    model = "resnet"
  elif exp_path.find("mobilenet") != -1:
    model = "mobilenet"
  elif exp_path.find("transformer") != -1:
    model = "transformer"
  else:
    exit(0)
  output_name = "ndpx_candidate_table_postprocessed.csv"
  output_path = os.path.join(exp_path, output_name)
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
    except:
      layer_parsed = original_elements[1]
    new_results = original_elements[0] + "," + \
                  layer_parsed + "," + \
                  original_elements[2] + "," + \
                  original_elements[3] + "\n"
    output.write(new_results)