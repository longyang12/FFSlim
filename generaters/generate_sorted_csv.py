import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/cc3m.csv')

# 提取数字并转换为整数类型
df['numeric_id'] = df['image_path']

# 按数字升序排序
sorted_df = df.sort_values(by='numeric_id', ascending=True)

# 生成新CSV文件（删除辅助列）
sorted_df.drop(columns=['numeric_id']).to_csv('/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/cc3m_sort.csv', index=False)