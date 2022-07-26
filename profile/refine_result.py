# refine the csv result.
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help="result csv file (.csv)")

if __name__ == "__main__":
  args = parser.parse_args()

  # read the result csv to dataframe
  result_dataframe = pd.read_csv(args.csv)
  result_dataframe['CYCLE'] = pd.to_numeric(result_dataframe['CYCLE'])

  intermediate_result = result_dataframe.groupby(['BREAKDOWN']).sum()[[ 'CYCLE']]
  print("before refining:")
  print(intermediate_result)
  print()

  # index: BREAKDOWN legends
  # row: CYCLE    2580219   Name: Fusion, dtype: int64
  for index, row in intermediate_result.iterrows():
    if "+" in index:
      print(index)
      legend_num = index.count("+") + 1
      legends = index.split("Fusion(")[1].split(")")[0].split("+")
      for legend in legends:
        legend_to_append = "Fusion(" + legend + ")"
        intermediate_result["CYCLE"][legend_to_append] += (row["CYCLE"] / legend_num)
      intermediate_result.drop(index, inplace=True)
  
  print()
  print("after refining:")
  print(intermediate_result)