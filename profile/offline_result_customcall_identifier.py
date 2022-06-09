# offline_execution_result.csv에 있는 custom-call instruction들에 대해서 알려줌
# custom call target을 가져옴
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--oe', type=str, help="offline_execution_result.csv", required=True)
parser.add_argument('--hlo', type=str, help="hlo_graph.txt", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  total_result = open(args.oe, 'r').read()
  hlo_graph = open(args.hlo, 'r').read()
  output_result = open("offline_execution_result_postprocessed.csv", "w+")

  total_results = total_result.split('\n')
  hlos = hlo_graph.split('\n')
  for result_row in total_results:
    if result_row.find("custom-call.") != -1:
      # custom call found
      custom_call_target = ""
      target = result_row.split(',')[0]
      # find the custom call
      for hlo in hlos:
        if (hlo.find(target) != -1) and (hlo.find("custom_call_target") != -1):
          custom_call_target = hlo.split('custom_call_target="')[1]
          custom_call_target = custom_call_target.split('"')[0]
          print(custom_call_target)
          result_row_data = result_row.split(",")
          print(result_row_data)
          result_row = result_row_data[0] + "$" + custom_call_target + "," + result_row_data[1]
          break
    print(result_row)
    output_result.write(result_row + "\n")