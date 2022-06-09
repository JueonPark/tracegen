# calculate the total offline execution result
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--oe", type=str, help="offline_execution_result_postprocessed.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  total_result = open(args.oe, 'r').read()

  total_results = total_result.split('\n', 1)[1]
  results = total_results.split('\n')
  total_time = 0
  for result in results[:-1]:
    if result.find("Ndp") != -1:
      continue
    total_time += (int)(result.split(',')[1])

  print(total_time)