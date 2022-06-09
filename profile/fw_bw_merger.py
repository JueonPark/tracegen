import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fw', type=str, help="fw.csv", required=True)
parser.add_argument('--bw', type=str, help="bw.csv", required=True)
parser.add_argument('--ffc', type=str, help="fw-full-cycle.csv", required=True)
parser.add_argument('--bfc', type=str, help="bw-full-cycle.csv", required=True)
parser.add_argument('--on', type=str, help="output name string", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  exp_path = os.getenv("EXP_PATH")

  # first, merge fw and bw
  total_result = open(args.fw, 'r').read()
  total_result += open(args.bw, 'r').read().split("\n", 1)[1]
  overall_output_name = args.on + ".csv"
  overall_output_path = os.path.join(exp_path, overall_output_name)
  overall_output = open(overall_output_path, 'w+')
  overall_output.write(total_result)

  # then, merge fw and bw with full cycle
  total_result = open(args.ffc, 'r').read()
  total_result += open(args.bfc, 'r').read().split("\n", 1)[1]
  overall_output_name = args.on + "-ndp-full-cycle.csv"
  overall_output_path = os.path.join(exp_path, overall_output_name)
  overall_output = open(overall_output_path, 'w+')
  overall_output.write(total_result)