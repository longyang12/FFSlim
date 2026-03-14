# 数据集下载
- flickr30k:https://huggingface.co/datasets/nlphuji/flickr30k/tree/main
- flickr30k_entities: https://github.com/BryanPlummer/flickr30k_entities
- mscoco:https://cocodataset.org/#download
- VQA2:https://huggingface.co/datasets/lmms-lab/VQAv2/tree/main/data
- GQA:https://huggingface.co/datasets/vikhyatk/gqa

<!-- # 预处理后的CSV文件（减少预处理时间：可选）
保存在阿里云盘：https://www.alipan.com/drive/home -->

# 数据集预处理
将不同形式的数据集转化为TDP格式，具体代码data_preprocess/*.py

# 数据集生成
将数据集转化为4种格式Files、TDP、FFRecord、FFSlim，具体代码 generaters/*.py

# 数据集加载
不同数据格式的数据集加载，具体代码 loaders/*.py

# 相关性能测试实验
测试不同数据格式的存储大小、内存大小、读写性能、转化开销等 experiment/*



