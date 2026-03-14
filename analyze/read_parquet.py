import pandas as pd
file_path = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/train_balanced-00001-of-00021.parquet"
df = pd.read_parquet(file_path, engine='pyarrow')
print(df.tail())
print(df.columns)
# print(df.describe())