import json

def display_json_structure(data, indent=0):
    """
    递归显示 JSON 数据的结构
    :param data: JSON 数据（字典或列表）
    :param indent: 当前缩进层级
    """
    prefix = "  " * indent  # 缩进
    if isinstance(data, dict):  # 如果是字典
        print(f"{prefix}{{")  # 打印字典起始
        for key, value in data.items():
            print(f"{prefix}  {key}: ", end="")
            if isinstance(value, (dict, list)):
                print()  # 换行后递归
                display_json_structure(value, indent + 2)
            else:
                print(f"{type(value).__name__}:{value}")  # 打印值的类型
        print(f"{prefix}}}")  # 打印字典结束
    elif isinstance(data, list):  # 如果是列表
        print(f"{prefix}[")  # 打印列表起始
        if len(data) > 0:
            for i, item in enumerate(data[:1]):  # 仅打印前 1 项，防止过长
                print(f"{prefix}  [{i}]: ", end="")
                if isinstance(item, (dict, list)):
                    print()
                    display_json_structure(item, indent + 2)
                else:
                    print(f"{type(item).__name__}:{item}")
            if len(data) > 1:
                print(f"{prefix}  ... ({len(data)} items in total)")
        print(f"{prefix}]")  # 打印列表结束
    else:  # 基本类型
        print(f"{type(data).__name__}:{data}")

# 读取 JSON 文件
file_path = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/v2_mscoco_train2014_annotations.json"  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 显示 JSON 结构
print("JSON 结构:")
display_json_structure(json_data)
