import pandas as pd

# 读取前1000行（自动包含表头）
df = pd.read_csv('/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/vqa2_sort.csv', nrows=1000)

# 保存到新文件（不保留索引）
df.to_csv('vqa2_sort_sample_1000.csv', index=False)